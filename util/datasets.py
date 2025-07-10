# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import random
from collections import defaultdict
import random
import PIL
import numpy as np
import torch
from PIL import ImageFilter, Image

from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split, Subset, DataLoader
from torchvision.datasets import ImageFolder, EuroSAT

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets import VOCSegmentation

from os.path import join
import pickle
import torchvision
from torch.utils.data import Dataset


class SubsetWithSamples(Subset):
    def __init__(self, subset) -> None:
        super().__init__(subset.dataset, subset.indices)

        if hasattr(subset.dataset, "samples"):
            self.samples = [subset.dataset.samples[i] for i in subset.indices]
        else:
            self.samples = None

        if hasattr(subset.dataset, "targets"):
            self.targets = [subset.dataset.targets[i] for i in subset.indices]
        else:
            self.targets = None


class CIFAR100Pickle(Dataset):
    def __init__(self, pickle_path, transform=None):
        self.transform = transform
        with open(pickle_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
            self.images = data_dict[b'data']
            self.labels = data_dict[b'fine_labels']

        # Reshape images (N, 3072) → (N, 3, 32, 32)
        self.images = self.images.reshape(-1, 32, 32, 3)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        img = torchvision.transforms.functional.to_pil_image(img)  # Convertir en PIL
        if self.transform:
            img = self.transform(img)
        return img, label


class TinyImagenetVal(ImageFolder):
    def find_classes(self, directory):
        classes, class_to_idx = super().find_classes(directory)
        # Filtrer "ORIG" s'il est présent
        classes = [c for c in classes if c != "ORIG"]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        return classes, class_to_idx


class TinyImageNetTestDataset(Dataset):
    """Custom dataset for Tiny ImageNet test set, as images are directly in a folder."""

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root) if fname.endswith(".JPEG")]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image  # No label for test set


class SiameseDataset(Dataset):
    """
    Wraps a classification dataset (like ImageFolder) to return two views of the same image
    along with the label, for use in Siamese or contrastive learning.
    """

    def __init__(self, base_dataset, transform):
        """
        Args:
            base_dataset: A dataset like torchvision.datasets.ImageFolder
            transform: A transform to apply separately to both views
        """
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        view1 = self.transform(image)
        view2 = self.transform(image)
        return (view1, view2), label


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    print(dataset)
    return dataset


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def get_transforms(input_size, siam_augment, dataset):

    if dataset == "tiny_imagenet":
        transform_train = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform_val = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])
        return transform_train, transform_val

    # simple augmentation
    if siam_augment:
        transform_train = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    transform_val = transforms.Compose([
        transforms.Resize(input_size, interpolation=3),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform_train, transform_val


def get_loaders(args, pretrain=True):
    transform_train, transform_val = get_transforms(args.input_size, args.siam_augment, args.dataset)

    if args.dataset == "caltech256":
        caltech256 = torchvision.datasets.Caltech256(
            root='../datasets/',
            download=True,
            transform=transform_train
        )
        generator = torch.Generator().manual_seed(42)
        train_size = int(0.7 * len(caltech256))
        val_size = int(0.15 * len(caltech256))
        test_size = len(caltech256) - train_size - val_size
        dataset_train, dataset_val, dataset_test = random_split(
            caltech256, [train_size, val_size, test_size], generator=generator
        )

    elif args.dataset == "tiny_imagenet":
        train_dir = "/data1/data/corpus/tiny-imagenet-200/train/"
        val_dir = "/data1/data/corpus/tiny-imagenet-200/val/"
        test_dir = "/data1/data/corpus/tiny-imagenet-200/test/images/"
        dataset_train = ImageFolder(root=train_dir, transform=transform_train)
        dataset_val = ImageFolder(root=val_dir, transform=transform_val)
        dataset_test = TinyImageNetTestDataset(root=test_dir, transform=transform_val)

    elif args.dataset == "cifar100":
        full_train_dataset = CIFAR100Pickle(pickle_path="../cifar-100-python/train", transform=transform_train)
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        generator = torch.Generator().manual_seed(42)
        dataset_train, dataset_val = random_split(full_train_dataset, [train_size, val_size], generator=generator)
        dataset_test = CIFAR100Pickle(pickle_path="../cifar-100-python/test", transform=transform_val)

    elif args.dataset in ["imagenette", "imagewoof"]:
        base_dir = f"../datasets/{args.dataset}2-320"
        train_dir = f"{base_dir}/train/"
        val_dir = f"{base_dir}/val/"

        # Validation and test split remain unchanged
        full_val_dataset = ImageFolder(root=val_dir, transform=transform_val)
        val_size = int(0.5 * len(full_val_dataset))
        test_size = len(full_val_dataset) - val_size
        generator = torch.Generator().manual_seed(42)
        dataset_val, dataset_test = random_split(full_val_dataset, [val_size, test_size], generator=generator)

        # Training dataset: standard or siamese
        base_train_dataset = ImageFolder(root=train_dir, transform=None)
        if pretrain and args.arch == "sjmae":
            # if args.w_siam > 0. or args.w_jigsaw > 0. and args.w_mae > 0.:
            if args.siam_augment:
                dataset_train = SiameseDataset(base_train_dataset, transform_train)
            else:
                dataset_train = ImageFolder(root=train_dir, transform=transform_train)
        else:
            dataset_train = ImageFolder(root=train_dir, transform=transform_train)


    elif args.dataset == "imagenet100":
        base_dir = "/media/jules/Transcend/Datasets/test_imgnet/"
        train_dir = f"{base_dir}/train/"
        val_dir = f"{base_dir}/val/"

        full_val_dataset = ImageFolder(root=val_dir, transform=transform_val)
        val_size = int(0.5 * len(full_val_dataset))
        test_size = len(full_val_dataset) - val_size
        generator = torch.Generator().manual_seed(42)
        dataset_val, dataset_test = random_split(full_val_dataset, [val_size, test_size], generator=generator)

        if args.label_ratio and args.label_ratio < 1.0:
            base_train_dataset = ImageFolder(root=train_dir, transform=transform_train)
            random.seed(42)
            indices_per_class = defaultdict(list)
            for idx, (_, label) in enumerate(base_train_dataset.samples):
                indices_per_class[label].append(idx)

            selected_indices = []
            for label, indices in indices_per_class.items():
                k = max(1, int(args.label_ratio * len(indices)))
                selected_indices.extend(random.sample(indices, k))

            base_train_dataset = torch.utils.data.Subset(base_train_dataset, selected_indices)
        else:
            base_train_dataset = ImageFolder(root=train_dir, transform=None)

        if pretrain and args.arch == "sjmae":
            # if args.w_siam > 0. or args.w_jigsaw > 0. and args.w_mae > 0.:
            if args.siam_augment:
                dataset_train = SiameseDataset(base_train_dataset, transform_train)
            else:
                dataset_train = ImageFolder(root=train_dir, transform=transform_train)
        else:
            if args.label_ratio and args.label_ratio < 1.0:
                dataset_train = base_train_dataset
                dataset_train.transform = transform_train
            else:
                dataset_train = ImageFolder(root=train_dir, transform=transform_train)

    elif args.dataset == "food101":
        base_dir = "../datasets/Food-101N/images/"
        train_dir = base_dir

        if args.label_ratio and args.label_ratio < 1.0:
            base_train_dataset = ImageFolder(root=train_dir, transform=transform_train)
        else:
            base_train_dataset = ImageFolder(root=train_dir, transform=None)

        train_size = int(0.7 * len(base_train_dataset))
        val_size = int(0.15 * len(base_train_dataset))
        test_size = len(base_train_dataset) - train_size - val_size
        generator = torch.Generator().manual_seed(42)
        dataset_train, _, _ = random_split(
            base_train_dataset, [train_size, val_size, test_size], generator=generator
        )
        _, dataset_val, dataset_test = random_split(base_train_dataset, [train_size, val_size, test_size],
                                                    generator=generator)
        dataset_val = SubsetWithSamples(dataset_val)
        dataset_test = SubsetWithSamples(dataset_test)
        dataset_val.dataset.transform = transform_val
        dataset_test.dataset.transform = transform_val
        dataset_train = SubsetWithSamples(dataset_train)
        dataset_train.dataset.transform = transform_train
        if args.label_ratio and args.label_ratio < 1.0:
            random.seed(42)

            indices_per_class = defaultdict(list)
            for idx, (_, label) in enumerate(dataset_train.samples):
                indices_per_class[label].append(idx)

            selected_indices = []
            for label, indices in indices_per_class.items():
                k = max(1, int(args.label_ratio * len(indices)))
                selected_indices.extend(random.sample(indices, k))

            dataset_train = torch.utils.data.Subset(dataset_train, selected_indices)

        if pretrain and args.arch == "sjmae":
            # if args.w_siam > 0. or args.w_jigsaw > 0. and args.w_mae > 0.:
            if args.siam_augment:
                dataset_train = SiameseDataset(dataset_train, transform_train)
            # else:
                # dataset_train = ImageFolder(root=train_dir, transform=transform_train)
        else:
            if args.label_ratio and args.label_ratio < 1.0:
                dataset_train = dataset_train
                dataset_train.transform = transform_train

    elif args.dataset == "vocseg":
        target_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.PILToTensor()  # For masks
        ])
        dataset_train = VOCSegmentation(
            root='data',
            year='2012',
            image_set='train',
            download=True,
            transform=transform_train,
            target_transform=target_transform
        )

        dataset_train = DataLoader(dataset_train, batch_size=8, shuffle=True)

        dataset_val = VOCSegmentation(
            root='data',
            year='2012',
            image_set='val',
            download=True,
            transform=transform_train,
            target_transform=target_transform
        )

        dataset_val = DataLoader(dataset_val, batch_size=8, shuffle=True)
        dataset_test = DataLoader(dataset_val, batch_size=8, shuffle=True)

    elif args.dataset == "eurosat":
        base_train_dataset = EuroSAT(
            root='../datasets/eurosat/',
            download=True,
            transform=transform_train,
        )

        train_size = int(0.7 * len(base_train_dataset))
        val_size = int(0.15 * len(base_train_dataset))
        test_size = len(base_train_dataset) - train_size - val_size
        generator = torch.Generator().manual_seed(42)
        dataset_train, _, _ = random_split(
            base_train_dataset, [train_size, val_size, test_size], generator=generator
        )
        _, dataset_val, dataset_test = random_split(base_train_dataset, [train_size, val_size, test_size],
                                                    generator=generator)
        dataset_val = SubsetWithSamples(dataset_val)
        dataset_test = SubsetWithSamples(dataset_test)
        dataset_val.dataset.transform = transform_val
        dataset_test.dataset.transform = transform_val
        dataset_train = SubsetWithSamples(dataset_train)
        dataset_train.dataset.transform = transform_train
        if args.label_ratio and args.label_ratio < 1.0:
            random.seed(42)

            indices_per_class = defaultdict(list)
            for idx, (_, label) in enumerate(dataset_train.samples):
                indices_per_class[label].append(idx)

            selected_indices = []
            for label, indices in indices_per_class.items():
                k = max(1, int(args.label_ratio * len(indices)))
                selected_indices.extend(random.sample(indices, k))

            dataset_train = torch.utils.data.Subset(dataset_train, selected_indices)

        if pretrain and args.arch == "sjmae":
            if args.siam_augment:
                dataset_train = SiameseDataset(dataset_train, transform_train)
        else:
            if args.label_ratio and args.label_ratio < 1.0:
                dataset_train = dataset_train
                dataset_train.transform = transform_train
    else:
        print("Unknown dataset!")
        exit(1)


    return dataset_train, dataset_val, dataset_test
