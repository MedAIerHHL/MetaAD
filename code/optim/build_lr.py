import math

import torch.nn as nn
from torch.optim import lr_scheduler
import timm
import timm.scheduler


def get_scheduler(optimizer, cfg):
    epochs = cfg.TRAIN.max_epoch
    if cfg.OPTIMIZER.lr_name == 'linear_lr':
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - cfg.OPTIMIZER.lr_min) + cfg.OPTIMIZER.lr_min
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    elif cfg.OPTIMIZER.lr_name == 'warmup_cosine_lr':
        scheduler = timm.scheduler.CosineLRScheduler(
            optimizer=optimizer,
            t_initial=cfg.TRAIN.max_epoch,
            lr_min=cfg.OPTIMIZER.lr_min,
            warmup_t=cfg.OPTIMIZER.warmup_epoch,
            warmup_lr_init=cfg.OPTIMIZER.lr_init,
        )

    # plot_lr_scheduler(optimizer, scheduler, epochs)
    return scheduler
