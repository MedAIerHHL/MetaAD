import argparse
import gc
from utils.tools import yaml_parser
from src.train_val_cyclegan import train_cyclegan


if __name__ == "__main__":
    gc.collect()

    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--yaml', default='./config/cyclegan.yaml', type=str, help='configuration filename')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter')
    parser.add_argument('--mode', default='train')
    args = parser.parse_args()
    cfg = yaml_parser(args.yaml)  # 解析配置文件

    train_cyclegan(cfg, args)