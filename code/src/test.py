import itertools
import os.path
import shutil
import time
import gc
import pandas as pd
import torch
from tqdm import tqdm

import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import optim

from utils.tools import *
from models import get_model
from data import get_dataloader
from loss import get_loss

def test_model(cfg, args):
    experiment_dir = os.path.join('../output', cfg.TEST.test_experiment_dir)
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    test_result_dir =  os.path.join(experiment_dir, 'test_result')
    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)
    iter_test = cfg.TEST.iter_test
    save_test_result_dir = os.path.join(test_result_dir, f"iter_{iter_test}")
    if not os.path.exists(save_test_result_dir):
        os.makedirs(save_test_result_dir)
    copyfile(args.yaml, str(save_test_result_dir) + '/' + os.path.basename(args.yaml))
    # ============================================= Initialize basic information =============================================
    # Loggings
    open_log(save_test_result_dir)
    logging.info("Start testing ......")
    logging.info(cfg)

    setup_seed(cfg.GLOBAL.manual_seed)

    # ----------------------------------------
    #      Initialize ENV and GPU
    # ----------------------------------------
    if cfg.GLOBAL.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif cfg.GLOBAL.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GLOBAL.device
        assert torch.cuda.is_available()

    cuda = cfg.GLOBAL.device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    rank, local_rank, world_size = get_envs()

    # DDP distriuted mode
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            init_method='env://',
            rank=local_rank,
            world_size=world_size)

    # GPU信息，清空显存
    GPU_info()

    # ----------------------------------------
    #      Initialize testing parameters
    # ----------------------------------------
    # Data
    val_data_path_list = (pd.read_csv(os.path.join(cfg.GLOBAL.data_file_dir, cfg.VAL.val_data_file))).iloc[:, -1].values.tolist()
    val_data_path_list = [os.path.join(cfg.GLOBAL.data_dir, i) for i in val_data_path_list]
    val_dataloader, num_val = get_dataloader(
        cfg,
        mode='val',
        mode_data_path_list=val_data_path_list,
        batch_size=cfg.VAL.batch_size,
        num_works=cfg.VAL.num_workers,
    )
    logging.info(f"num_val: {num_val}")

    test_data_path_list = (pd.read_csv(os.path.join(cfg.GLOBAL.data_file_dir, cfg.TEST.test_data_file))).iloc[:, -1].values.tolist()
    test_data_path_list = [os.path.join(cfg.GLOBAL.data_dir, i) for i in test_data_path_list]
    test_dataloader, num_test = get_dataloader(
        cfg,
        mode='test',
        mode_data_path_list=test_data_path_list,
        batch_size=cfg.TEST.batch_size,
        num_works=cfg.TEST.num_workers,
    )
    logging.info(f"num_test: {num_test}")

    # Model
    if cfg.TEST.iter_test == 'best':
        models = get_model(cfg, )
    elif cfg.TEST.iter_test:
        models = get_model(cfg, iter=cfg.TEST.iter_test)
    else:
        logging.info("Please set cfg.TEST.iter_model or cfg.TEST.iter_test")

    vae = models[0].to(device)
    net_D = models[1].to(device)
    vae.eval()
    net_D.eval()

    vae_loss_fn = get_loss('VAELoss')

    # ============================================= Testing =============================================
    data_loader_list = {'val': val_dataloader, 'test': test_dataloader}
    num_data_list = {'val': num_val, 'test': num_test}
    for mode, data_loader in data_loader_list.items():
        num_data = num_data_list[mode]
        logging.info(f"Start testing {mode} data, total {num_data} images ......")
        with torch.no_grad():
            for _, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="Testing"):
                real_images = data['PET_img'].cuda().float()
                img_name = data["img_name"][0]
                recon_images, real_images, mu, logvar = vae(real_images)
                mode_loss_outputs = vae_loss_fn([recon_images, real_images, mu, logvar], kld_weight=cfg.LOSS.kld_weight)
                mode_total_loss = mode_loss_outputs['loss']
                mode_total_loss += mode_total_loss.item()
                if cfg.TEST.save_test_result:
                    save_vol_image(real_images, img_name, os.path.join(save_test_result_dir, "input_FDG"))
                    save_vol_image(recon_images, img_name, os.path.join(save_test_result_dir, "recon_FDG"))
                    recon_FDG_error_map = get_error_map(recon_images, real_images)
                    save_vol_image(recon_FDG_error_map, img_name, os.path.join(save_test_result_dir, "recon_FDG_error_map"))
            mode_total_loss /= len(val_dataloader)
            logging.info(f"Total loss of {mode} data is {mode_total_loss}")
    logging.info("Finish testing ......")

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--yaml', default='./config/vae.yaml', type=str, help='output model name')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    gc.collect()
    args = get_args()   # 读取入参
    cfg = yaml_parser(args.yaml)  # 解析配置文件
    # experiment_dir, checkpoints_dir, tensorboard_dir = init_setting(cfg, args)
    # logger(experiment_dir, os.path.basename(experiment_dir))
    # shutil.copyfile('main.py', str(experiment_dir.joinpath('setting/')) + '/' + 'main.py')
    test_model(cfg, args)


