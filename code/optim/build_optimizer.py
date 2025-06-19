import torch.nn as nn
from torch.optim import Adam, SGD

def get_optimizer(parameters, cfg):
    if cfg.OPTIMIZER.name == 'sgd':
        optimizer = SGD(
            params=parameters,
            lr=cfg.OPTIMIZER.lr_max,
            momentum=cfg.OPTIMIZER.momentum,
            weight_decay=cfg.OPTIMIZER.weight_decay,
        )
    elif cfg.OPTIMIZER.name == 'adam':
        optimizer = Adam(
            params=parameters,
            lr=cfg.OPTIMIZER.lr_g,
            betas=(cfg.OPTIMIZER.beta1, cfg.OPTIMIZER.beta2)
        )
    return optimizer