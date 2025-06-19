import argparse
import gc
from utils.tools import yaml_parser
from src.test_cyclegan import test_cyclegan
from src.test_unet import test_unet


if __name__ == "__main__":
    gc.collect()

    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--yaml', default='./config/cyclegan.yaml', type=str, help='configuration filename')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter')
    parser.add_argument('--mode', default='test')
    args = parser.parse_args()
    cfg = yaml_parser(args.yaml)  # 解析配置文件
    print(args.mode)
    test_cyclegan(cfg, args)