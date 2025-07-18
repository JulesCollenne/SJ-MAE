# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
from typing import Iterable

import torch

import util.lr_sched as lr_sched
import util.misc as misc


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if isinstance(samples, (list, tuple)):
            samples = [s.to(device, non_blocking=True) for s in samples]
        else:
            samples = samples.to(device, non_blocking=True)

        # with torch.cuda.amp.autocast():
        with torch.amp.autocast(args.device):
            loss, _, _, losses = model(samples)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}".format(loss_value))
            print(losses)
            # sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if args.device == 'cuda':
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_recon=losses.recon)
        if args.w_jigsaw > 0.:
            metric_logger.update(loss_jigsaw=losses.jigsaw)
        if args.w_siam > 0.:
            metric_logger.update(loss_siam=losses.siam)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_recon_reduce = misc.all_reduce_mean(losses.recon)
        if args.w_jigsaw > 0.:
            loss_jigsaw_reduce = misc.all_reduce_mean(losses.jigsaw)
        if args.w_siam > 0.:
            loss_siam_reduce = misc.all_reduce_mean(losses.siam)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_recon', loss_recon_reduce, epoch_1000x)
            if args.w_jigsaw > 0.:
                log_writer.add_scalar('train_loss_jigsaw', loss_jigsaw_reduce, epoch_1000x)
            if args.w_siam > 0.:
                log_writer.add_scalar('train_loss_siam', loss_siam_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

