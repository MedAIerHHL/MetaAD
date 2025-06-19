import argparse
import gc
from utils.tools import yaml_parser
from src.train_val_cyclegan import train_cyclegan
from src.train_val_unet import train_unet
# from src.train_val_cyclegan_loss_cv import train_cyclegan_loss
# from src.train_val_cyclegan_loss import train_cyclegan_loss
from src.train_val_cyclegan_loss_1 import train_cyclegan_loss


if __name__ == "__main__":
    gc.collect()

    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--yaml', default='./config/cyclegan.yaml', type=str, help='configuration filename')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter')
    parser.add_argument('--mode', default='train')
    args = parser.parse_args()
    cfg = yaml_parser(args.yaml)  # 解析配置文件
    if 'loss' in cfg.GLOBAL.exp_name:
        train_cyclegan_loss(cfg, args)
    elif ('unet' in cfg.GLOBAL.exp_name) or ('ae' in cfg.GLOBAL.exp_name):
        train_unet(cfg, args)
    else:
        train_cyclegan(cfg, args)
        train_cyclegan()