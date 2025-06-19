import sys
import torch

def get_loss(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_class = (torch.tensor(cfg.LOSS.weight_class, dtype=torch.float32)).to(device)
    if cfg.LOSS.name == 'CELoss':
        loss_function = torch.nn.CrossEntropyLoss(weight= weight_class)
    else:
        loss_function = None
        print('Warning: unknown loss function.')
        sys.exit(0)
    return loss_function

