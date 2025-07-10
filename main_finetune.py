# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import datetime
import json
import os
import time
from os.path import join
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.layers import trunc_normal_
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

import models_vit
import util.lr_decay as lrd
import util.misc as misc
from engine_finetune import train_one_epoch, evaluate
from util.datasets import get_loaders
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.parser import parse_arguments, read_config_file, extract_mask_ratios_from_path
from util.pos_embed import interpolate_pos_embed


def get_param_groups(model, weight_decay, layer_decay):
    decay = []
    no_decay = []

    # Layer-wise decay is generally not used for ResNets, but if you want, you need to define layer groups manually.
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bn' in name or 'bias' in name:  # No weight decay for batch norm & biases
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0}
    ]

def main(args):
    args.distributed = False
    args.global_rank = 0
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train, dataset_val, dataset_test = get_loaders(args, False)

    global_rank = 0

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.arch == "sjmae":
        if args.resume:
            args.mask_ratio_mae, args.mask_ratio_jigsaw, args.w_jigsaw, args.w_siam = extract_mask_ratios_from_path(
                args.resume)
        elif args.finetune:
            args.mask_ratio_mae, args.mask_ratio_jigsaw, args.w_jigsaw, args.w_siam = extract_mask_ratios_from_path(
                args.finetune)
        else:
            print("ERROR WHEN GETTING PARAMS")
            exit(1)
        args.output_dir += args.arch + "/" + args.dataset + f"/mask_ratio_{str(args.mask_ratio_mae)}_{args.mask_ratio_jigsaw}/" + f"w_{str(args.w_jigsaw)}_{str(args.w_siam)}" + "/finetune/"
    else:
        args.output_dir += args.arch + "/" + args.dataset + "/finetune/"
    args.log_dir = args.output_dir + "/logs/"
    
    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        logpath = join(args.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        log_writer = SummaryWriter(log_dir=logpath)
        file_path = join(logpath, 'args.txt')
        formatted_output = "{}".format(args).replace(', ', ',\n')
        with open(file_path, 'w') as file:
            file.write(formatted_output)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        img_size=args.input_size
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    if args.global_pool:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    else:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    # manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)


    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                    no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                    layer_decay=args.layer_decay
                                    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()
    class_weights = None
    # class_weights = torch.tensor([0.02, 0.98], dtype=torch.float32).to(device)
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    print("criterion = %s" % str(criterion))
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)
    if args.test:
        run_final_test_evaluation(data_loader_test, model, device, args)
        exit(0)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % args.ckp_interval == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)
            if args.nb_classes == 2:
                log_writer.add_scalar('perf/test_auc', test_stats['auc'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

    if args.output_dir and misc.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
        # ---- FINAL TEST SET EVALUATION ----
    run_final_test_evaluation(data_loader_test, model, device, args)



def run_final_test_evaluation(data_loader_test, model, device, args):
    print("\n🔍 Final evaluation on the test set...")

    test_stats = evaluate(data_loader_test, model, device)

    print(f"\n🎯 Final Test Accuracy (Top-1): {test_stats['acc1']:.2f}%")
    print(f"🎯 Final Test Accuracy (Top-5): {test_stats['acc5']:.2f}%")
    if args.nb_classes == 2:
        print(f"🧬 Final Test AUC: {test_stats['auc']:.2f}%")

    # Save to file
    results_path = os.path.join(args.output_dir, "final_test_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Test Results for {args.arch} on {args.dataset}\n")
        f.write(f"Top-1 Accuracy: {test_stats['acc1']:.2f}%\n")
        f.write(f"Top-5 Accuracy: {test_stats['acc5']:.2f}%\n")
        if args.nb_classes == 2:
            f.write(f"AUC: {test_stats['auc']:.2f}%\n")
        f.write("\n--- Full Args ---\n")
        f.write(str(args))

    print(f"✅ Final test results saved to {results_path}")



if __name__ == '__main__':
    cfg_arg = parse_arguments()
    arch = cfg_arg.config.split("/")[-3]
    args = read_config_file(cfg_arg.config, default_file=f"./configs/{arch}/finetune/default.yaml")
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

