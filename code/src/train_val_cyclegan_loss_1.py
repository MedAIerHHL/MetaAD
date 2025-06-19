import itertools
import logging
import os.path
import shutil
import time
import gc

from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.utils.data.dataset import ConcatDataset

from loss.gan_loss import GANLoss
from utils.tools import *
from utils.metrics import *
from utils.viz import *
from code.data.datasets.pet_dataset_2 import PETDataset, InfiniteSamplerWrapper
from loss import get_loss
from optim.build_optimizer import get_optimizer
from models.cyclegan.utils import create_generator, create_discriminator

def gen_random_mask(x, mask_k, mask_size, mask_prob):
    if random.uniform(0, 1) <= mask_prob:
        b, c, d, h, w = x.shape
        L = (d // mask_size) * (h // mask_size) * (w // mask_size)
        len_keep = int(L * (1-mask_k))

        noise = torch.randn(b, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([b, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = mask.reshape(-1, d // mask_size, h // mask_size, w // mask_size). \
            repeat_interleave(mask_size, dim=1). \
            repeat_interleave(mask_size, dim=2). \
            repeat_interleave(mask_size, dim=3)
        mask = mask.unsqueeze(1).type_as(x)
        x = x * (1. - mask) - mask
    return x
def get_random_subregion(image):
    d = np.random.randint(8, int(image.shape[0]//3))
    h = np.random.randint(8, int(image.shape[1]//3))
    w = np.random.randint(8, int(image.shape[2]//3))
    d_start = np.random.randint(12, image.shape[0] - d -12)
    h_start = np.random.randint(16, image.shape[1] - h -16)
    w_start = np.random.randint(14, image.shape[2] - w -14)
    return [d, h, w], [d_start, h_start, w_start]
def add_noise_to_subregion(image, noise_type, noise_scale, sampling_rate, intensity=1, mean=0, std=0.2):
    size, start = get_random_subregion(image)
    subregion = image[start[0]:start[0]+size[0], start[1]:start[1]+size[1], start[2]:start[2]+size[2]]
    noise_res = [i // sampling_rate for i in size]
    if noise_type == 'poisson':
        noise = torch.poisson(intensity * torch.ones(noise_res[0], noise_res[1], noise_res[2])).to(subregion.device)
    elif noise_type == 'gaussian':
        noise = torch.normal(mean, std, size=(noise_res[0], noise_res[1], noise_res[2])).to(subregion.device)
    else:
        raise ValueError('Invalid noise type')
    # min_val = noise.min()
    # max_val = noise.max()
    # noise = 2 * (noise - min_val) / (max_val - min_val) - 1
    noise = noise.unsqueeze(0).unsqueeze(0)
    noise = nn.functional.interpolate(noise, size = size, mode='trilinear')
    noise = noise.squeeze(0).squeeze(0)
    roll_x = random.choice(range(size[0]))
    roll_y = random.choice(range(size[1]))
    roll_z = random.choice(range(size[2]))
    noise = torch.roll(noise, shifts=[roll_x, roll_y, roll_z], dims=[-3, -2, -1])
    # mask = subregion.sum(dim=0, keepdim=True) > 0.01
    # noise *= mask
    noisy_subregion = subregion + noise * noise_scale
    image[start[0]:start[0]+size[0], start[1]:start[1]+size[1], start[2]:start[2]+size[2]] = noisy_subregion
    return image
def add_noise_to_batch(batch, noise_type, noise_scale, sampling_rate, intensity=1, mean=0, std=1):
    for b in range(batch.shape[0]):
        for c in range(batch.shape[1]):
            batch[b, c] = add_noise_to_subregion(batch[b, c], noise_type, noise_scale, sampling_rate, intensity, mean, std)
    return batch

def train_cyclegan_loss(cfg, args):
    model_name = cfg.MODEL.name
    if model_name == 'cyclegan_loss_noise':
        assert cfg.MODEL.noise == True and cfg.MODEL.mask == False, 'Invalid model configuration'
    elif model_name == 'cyclegan_loss_mask':
        assert cfg.MODEL.noise == False and cfg.MODEL.mask == True, 'Invalid model configuration'
    else:
        ValueError('Invalid model name')
    # ============================================= Initialize basic information =============================================
    # The root directory of the training experiment, the directory where the model is saved, and the directory where the log is saved
    experiment_dir, checkpoints_dir, tensorboard_dir = init_setting(cfg, args)
    # Loggings
    open_log(experiment_dir, log_name='train_val')
    logging.info(os.path.abspath(experiment_dir))
    logging.info(os.path.abspath(checkpoints_dir))
    logging.info(os.path.abspath(tensorboard_dir))
    logging.info(cfg)

    setup_seed(cfg.GLOBAL.manual_seed)

    tb_writer = SummaryWriter(tensorboard_dir)

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
    #      data parallel
    # ----------------------------------------

    # Data
    # Define training dataset
    train_data = PETDataset(
        cfg.GLOBAL.data_dir,
        cfg.TRAIN.train_data_file,
        cfg.TRAIN.image_size,
        cfg.TRAIN.data_norm,
    )
    logging.info('The overall number of training images is %d' % len(train_data))

    train_loader = iter(DataLoader(
        train_data,
        batch_size = cfg.TRAIN.batch_size,
        drop_last = True,
        sampler = InfiniteSamplerWrapper(train_data),
        num_workers = cfg.TRAIN.num_workers,)
    )


    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------
    resume_checkpoint = cfg.MODEL.resume_checkpoint
    # Build generators
    if args.mode == 'train' and cfg.MODEL.pretrain_gen:
        netG_name1 = os.path.join(resume_checkpoint, 'gen_to_CFT_{:d}.pth'.format(cfg.MODEL.pretrain_gen))
        netG_name2 = os.path.join(resume_checkpoint, 'gen_to_FDG_{:d}.pth'.format(cfg.MODEL.pretrain_gen))
    else:
        netG_name1 = None
        netG_name2 = None
    netG_to_CFT = create_generator(cfg, netG_name1).cuda()
    netG_to_FDG = create_generator(cfg, netG_name2).cuda()

    # Build discriminators
    if args.mode == 'train' and cfg.MODEL.pretrain_dis:
        netD_name1 = os.path.join(resume_checkpoint, 'dis_to_CFT_{:d}.pth'.format(cfg.MODEL.pretrain_dis))
        netD_name2 = os.path.join(resume_checkpoint, 'dis_to_FDG_{:d}.pth'.format(cfg.MODEL.pretrain_dis))
    else:
        netD_name1 = None
        netD_name2 = None
    netD_FDG = create_discriminator(cfg, netD_name1).cuda()
    netD_CFT = create_discriminator(cfg, netD_name2).cuda()

    # Build optimizer
    optimizer_netG = get_optimizer(
        itertools.chain(netG_to_CFT.parameters(), netG_to_FDG.parameters()),
        cfg,
    )
    optimizer_netD = get_optimizer(
        itertools.chain(netD_CFT.parameters(), netD_FDG.parameters()),
        cfg,
    )

    # loss
    GAN_criterion = GANLoss(gan_mode=cfg.LOSS.gan_mode).cuda()
    # contrast_loss = torch.nn.CosineEmbeddingLoss()
    contrast_loss = NegativeL1Loss()

    # optionally resume from a checkpoint
    # Load pretrained models
    if cfg.MODEL.pretrain_gen:
        start_iter = cfg.MODEL.pretrain_gen
        logging.info("Resume training from iteration %d" % start_iter)
    else:
        start_iter = 0
        logging.info("Start training ......")

    min_delta = cfg.TRAIN.early_stop_min_delta
    early_stop_patience = cfg.TRAIN.early_stop_patience
    early_stop = False
    early_stopping = EarlyStopping(patience=early_stop_patience, min_delta=min_delta)
    
    # 最佳模型的性能初始化
    best_iter = 0
    # best_iter_wt = deepcopy(model.state_dict())
    best_iter_eval_metric = 0.0
    best_iter_auc = 0.0
    best_iter_thresh = 0.5
    best_iter_acc = 0.0
    best_iter_spe = 0.0
    best_iter_sen = 0.0
    best_iter_f1 = 0.0
    best_iter_perfs = {}
    save_sample = False
    # ============================================= Train-Val =============================================
    since = time.time()
    for iter_index in tqdm(range(start_iter, cfg.TRAIN.max_iter), desc='Training'):
        cur_iter = iter_index + 1
        # ============================================= Train =============================================
        # Load images ([B, 1, D, H, W])
        data = next(train_loader)
        real_FDG_img, real_CFT_img = data["FDG_img"].cuda().float(), data["CFT_img"].cuda().float()
        paired_flag = data["paired_flag"]

        if model_name == 'cyclegan_loss_noise':
            noise_real_FDG_img, noise_real_CFT_img = \
                add_noise_to_batch(data["FDG_img"].float(), cfg.MODEL.noise_type, cfg.MODEL.noise_scale, cfg.MODEL.sampling_rate), \
                    add_noise_to_batch(data["CFT_img"].float(), cfg.MODEL.noise_type, cfg.MODEL.noise_scale, cfg.MODEL.sampling_rate)
        elif model_name == 'cyclegan_loss_mask':
            noise_real_FDG_img, noise_real_CFT_img = \
                gen_random_mask(data["FDG_img"].float(), mask_k=cfg.MODEL.mask_k, mask_size=cfg.MODEL.mask_size, mask_prob=cfg.MODEL.mask_prob), \
                    gen_random_mask(data["CFT_img"].float(), mask_k=cfg.MODEL.mask_k, mask_size=cfg.MODEL.mask_size, mask_prob=cfg.MODEL.mask_prob)

        noise_real_FDG_img, noise_real_CFT_img = noise_real_FDG_img.cuda(), noise_real_CFT_img.cuda()
        if not save_sample:
            real_FDG_img_example = real_FDG_img[0].clone()
            noise_real_FDG_img_example = noise_real_FDG_img[0].clone()
            save_vol_image(real_FDG_img_example, 'real_FDG_img_example.nii', os.path.join(experiment_dir, 'samples_viz'), cfg.TRAIN.data_norm)
            save_vol_image(noise_real_FDG_img_example, 'noise_real_FDG_img_example.nii', os.path.join(experiment_dir, 'samples_viz'), cfg.TRAIN.data_norm)
            real_CFT_img_example = real_CFT_img[0].clone()
            noise_real_CFT_img_example = noise_real_CFT_img[0].clone()
            save_vol_image(real_CFT_img_example, 'real_CFT_img_example.nii', os.path.join(experiment_dir, 'samples_viz'), cfg.TRAIN.data_norm)
            save_vol_image(noise_real_CFT_img_example, 'noise_real_CFT_img_example.nii', os.path.join(experiment_dir, 'samples_viz'), cfg.TRAIN.data_norm)
            save_sample == True

        # ----------------------------------------
        #            Train Generators
        # ----------------------------------------

        # FDG --> CFT --> FDG
        # fake_CFT_img = netG_to_CFT(real_FDG_img)
        # recon_FDG_img = netG_to_FDG(fake_CFT_img)
        # noise_fake_CFT_img = netG_to_CFT(noise_real_FDG_img)
        # noise_recon_FDG_img = netG_to_FDG(noise_fake_CFT_img)
        fake_CFT_img = netG_to_CFT(noise_real_FDG_img)
        recon_FDG_img = netG_to_FDG(fake_CFT_img)
        noise_fake_CFT_img = netG_to_CFT(noise_real_FDG_img)
        noise_recon_FDG_img = netG_to_FDG(noise_fake_CFT_img)
        # with torch.no_grad:
        #     anormal_fake_CFT_img = netG_to_CFT(anormal_real_FDG_img)

        # CFT --> FDG --> CFT
        # fake_FDG_img = netG_to_FDG(real_CFT_img)
        # recon_CFT_img = netG_to_CFT(fake_FDG_img)
        # noise_fake_FDG_img = netG_to_FDG(noise_real_CFT_img)
        # noise_recon_CFT_img = netG_to_CFT(noise_fake_FDG_img)
        fake_FDG_img = netG_to_FDG(noise_real_CFT_img)
        recon_CFT_img = netG_to_CFT(fake_FDG_img)
        noise_fake_FDG_img = netG_to_FDG(noise_real_CFT_img)
        noise_recon_CFT_img = netG_to_CFT(noise_fake_FDG_img)

        # identity mapping
        same_FDG_img = netG_to_FDG(real_FDG_img)
        same_CFT_img = netG_to_CFT(real_CFT_img)

        # same_FDG_img2 = netG_to_FDG(real_FDG_img)
        # same_CFT_img2 = netG_to_CFT(real_CFT_img)

        # (0) mid contrast
        mid_contrast_loss_CFT = F.l1_loss(fake_CFT_img, noise_fake_CFT_img) * cfg.LOSS.mid_contrast_weight
        mid_contrast_loss_FDG = F.l1_loss(fake_FDG_img, noise_fake_FDG_img) * cfg.LOSS.mid_contrast_weight
        mid_contrast_loss = mid_contrast_loss_CFT + mid_contrast_loss_FDG

        # (1) compute cycle loss
        cycle_loss_FDG = F.l1_loss(recon_FDG_img, real_FDG_img) * cfg.LOSS.cycle_weight
        cycle_loss_CFT = F.l1_loss(recon_CFT_img, real_CFT_img) * cfg.LOSS.cycle_weight
        cycle_loss = cycle_loss_FDG + cycle_loss_CFT

        # (2) compute GAN loss
        if cfg.TRAIN.gan_start_iter and cur_iter >= cfg.TRAIN.gan_start_iter:
            fake_FDG_out = netD_FDG(fake_FDG_img)
            fake_CFT_out = netD_CFT(fake_CFT_img)
            GAN_loss_G = ((GAN_criterion(fake_CFT_out, True, for_discriminator=False) +
                          GAN_criterion(fake_FDG_out, True, for_discriminator=False)) *
                          cfg.LOSS.gan_weight)
        else:
            GAN_loss_G = 0

        # (3) compute identity loss
        identity_loss_FDG = F.l1_loss(same_FDG_img, real_FDG_img) * cfg.LOSS.identity_weight
        identity_loss_CFT = F.l1_loss(same_CFT_img, real_CFT_img) * cfg.LOSS.identity_weight
        identity_loss = identity_loss_FDG + identity_loss_CFT

        # (4) compute noise cycle loss
        if model_name == 'cyclegan_loss_noise':
            if cfg.LOSS.cycle_noise_type == 'l1':
                noise_cycle_loss_FDG = F.l1_loss(noise_recon_FDG_img, real_FDG_img) * cfg.LOSS.cycle_noise_weight
                noise_cycle_loss_CFT = F.l1_loss(noise_recon_CFT_img, real_CFT_img) * cfg.LOSS.cycle_noise_weight
            elif cfg.LOSS.cycle_noise_type == 'l2':
                noise_cycle_loss_FDG = F.mse_loss(noise_recon_FDG_img, real_FDG_img) * cfg.LOSS.cycle_noise_weight
                noise_cycle_loss_CFT = F.mse_loss(noise_recon_CFT_img, real_CFT_img) * cfg.LOSS.cycle_noise_weight
        elif model_name == 'cyclegan_loss_mask':
            noise_cycle_loss_FDG = F.l1_loss(noise_recon_FDG_img, real_FDG_img) * cfg.LOSS.cycle_noise_weight
            noise_cycle_loss_CFT = F.l1_loss(noise_recon_CFT_img, real_CFT_img) * cfg.LOSS.cycle_noise_weight
        noise_cycle_loss = noise_cycle_loss_FDG + noise_cycle_loss_CFT

        # backward and optimize
        total_loss_G = mid_contrast_loss + cycle_loss + GAN_loss_G + identity_loss + noise_cycle_loss
        optimizer_netG.zero_grad()
        total_loss_G.backward()
        optimizer_netG.step()

        # ----------------------------------------
        #          Train Discriminators
        # ----------------------------------------

        if cfg.TRAIN.gan_start_iter and cur_iter >= cfg.TRAIN.gan_start_iter:
            # (1) compute loss with real images
            real_FDG_out = netD_FDG(real_FDG_img)
            real_CFT_out = netD_CFT(real_CFT_img)
            GAN_loss_D_real = GAN_criterion(real_FDG_out, True, for_discriminator=True) + \
                              GAN_criterion(real_CFT_out, True, for_discriminator=True)

            # (2) compute loss with fake images
            with torch.no_grad():
                fake_CFT_img = netG_to_CFT(real_FDG_img)
                fake_FDG_img = netG_to_FDG(real_CFT_img)
            fake_CFT_out = netD_CFT(fake_CFT_img)
            fake_FDG_out = netD_FDG(fake_FDG_img)
            GAN_loss_D_fake = GAN_criterion(fake_CFT_out, False, for_discriminator=True) + \
                              GAN_criterion(fake_FDG_out, False, for_discriminator=True)

            # (3) backward and optimize
            GAN_loss_D = (GAN_loss_D_real + GAN_loss_D_fake) * 0.5 * cfg.LOSS.gan_weight
            optimizer_netD.zero_grad()
            GAN_loss_D.backward()
            optimizer_netD.step()

        # ----------------------------------------
        #            Log training states
        # ----------------------------------------
        # 把上面的loss全部用tb_writer保存
        tb_writer.add_scalar('Train/GAN_loss_G', GAN_loss_G.item()/cfg.LOSS.gan_weight, cur_iter)
        tb_writer.add_scalar('Train/GAN_loss_D', GAN_loss_D.item()/cfg.LOSS.gan_weight, cur_iter)
        # tb_writer.add_scalar('Train/mid_contrast_loss_FDG', mid_contrast_loss_FDG.item()/cfg.LOSS.mid_contrast_weight, cur_iter)
        # tb_writer.add_scalar('Train/mid_contrast_loss_CFT', mid_contrast_loss_CFT.item()/cfg.LOSS.mid_contrast_weight, cur_iter)
        # tb_writer.add_scalar('Train/mid_contrast_loss', mid_contrast_loss.item()/cfg.LOSS.mid_contrast_weight, cur_iter)
        tb_writer.add_scalar('Train/cycle_loss_FDG', cycle_loss_FDG.item()/cfg.LOSS.cycle_weight, cur_iter)
        tb_writer.add_scalar('Train/cycle_loss_CFT', cycle_loss_CFT.item()/cfg.LOSS.cycle_weight, cur_iter)
        tb_writer.add_scalar('Train/cycle_loss', cycle_loss.item()/cfg.LOSS.cycle_weight, cur_iter)
        tb_writer.add_scalar('Train/identity_loss_FDG', identity_loss_FDG.item()/cfg.LOSS.identity_weight, cur_iter)
        tb_writer.add_scalar('Train/identity_loss_CFT', identity_loss_CFT.item()/cfg.LOSS.identity_weight, cur_iter)
        tb_writer.add_scalar('Train/identity_loss', identity_loss.item()/cfg.LOSS.identity_weight, cur_iter)
        # tb_writer.add_scalar('Train/recon_loss_CFT', recon_loss_CFT.item()/cfg.LOSS.recon_weight, cur_iter)
        # tb_writer.add_scalar('Train/recon_loss_FDG', recon_loss_FDG.item()/cfg.LOSS.recon_weight, cur_iter)
        # tb_writer.add_scalar('Train/recon_loss', recon_loss.item()/cfg.LOSS.recon_weight, cur_iter)
        # tb_writer.add_scalar('Train/noise_cycle_loss_FDG', noise_cycle_loss_FDG.item()/cfg.LOSS.cycle_noise_weight, cur_iter)
        # tb_writer.add_scalar('Train/noise_cycle_loss_CFT', noise_cycle_loss_CFT.item()/cfg.LOSS.cycle_noise_weight, cur_iter)
        # tb_writer.add_scalar('Train/noise_cycle_loss', noise_cycle_loss.item()/cfg.LOSS.cycle_noise_weight, cur_iter)
        tb_writer.add_scalar('Train/contrast_loss_FDG', contrast_loss_FDG.item(), cur_iter)
        tb_writer.add_scalar('Train/contrast_loss_CFT', contrast_loss_CFT.item(), cur_iter)
        tb_writer.add_scalar('Train/contrast_loss', contrast_loss_all.item(), cur_iter)
        tb_writer.add_scalar('Train/total_loss_G', total_loss_G.item(), cur_iter)
        
        if cfg.TRAIN.log_interval and cur_iter % cfg.TRAIN.log_interval == 0:
            # print training status
            logging.info(
                '[Iteration {:d}] CFT Identity Loss: {:.4f} | FDG Identity Loss: {:.4f}'
                .format(cur_iter, identity_loss_CFT.item()/cfg.LOSS.identity_weight, identity_loss_FDG.item()/cfg.LOSS.identity_weight)
            )

            if cfg.TRAIN.gan_start_iter and cur_iter >= cfg.TRAIN.gan_start_iter:
                with torch.no_grad():
                    real_CFT_score = real_CFT_out.mean().item()
                    fake_CFT_score = fake_CFT_out.mean().item()
                    real_FDG_score = real_FDG_out.mean().item()
                    fake_FDG_score = fake_FDG_out.mean().item()
                # print training status
                logging.info(
                    '[Iteration {:d} | FDG -> CFT -> FDG] Cycle Loss: {:.4f} | D(y) / D(G(x)): {:.4f} / {:.4f}'
                    .format(cur_iter, cycle_loss_FDG.item()/cfg.LOSS.cycle_weight, real_CFT_score, fake_CFT_score)
                )
                logging.info(
                    '[Iteration {:d} | CFT -> FDG -> CFT] Cycle Loss: {:.4f} | D(x) / D(G(y)): {:.4f} / {:.4f}'
                    .format(cur_iter, cycle_loss_CFT.item()/cfg.LOSS.cycle_weight, real_FDG_score, fake_FDG_score)
                )

            else:
                # print training status
                logging.info(
                    '[Iteration {:d} | FDG -> CFT -> FDG] Cycle Loss: {:.4f}'.format(cur_iter, cycle_loss_FDG.item()/cfg.LOSS.cycle_weight)
                )
                logging.info(
                    '[Iteration {:d} | CFT -> FDG -> CFT] Cycle Loss: {:.4f}'.format(cur_iter, cycle_loss_CFT.item()/cfg.LOSS.cycle_weight)
                )
        # ============================================= Validation =============================================
        if (cur_iter % cfg.TRAIN.val_interval == 0 and cur_iter >= cfg.TRAIN.val_interval) or cur_iter == cfg.TRAIN.max_iter:
            # 1. save models
            # (1) save generators
            save_model(netG_to_CFT, 'gen_to_CFT_{:d}.pth'.format(cur_iter), checkpoints_dir)
            save_model(netG_to_FDG, 'gen_to_FDG_{:d}.pth'.format(cur_iter), checkpoints_dir)
            logging.info('The generators are successfully saved at iteration {:d}'.format(cur_iter))
            # (2) Save discriminators
            if cfg.TRAIN.gan_start_iter and cur_iter >= cfg.TRAIN.gan_start_iter:
                save_model(netD_CFT, 'dis_to_CFT_{:d}.pth'.format(cur_iter), checkpoints_dir)
                save_model(netD_FDG, 'dis_to_FDG_{:d}.pth'.format(cur_iter), checkpoints_dir)
                logging.info('The discriminators are successfully saved at iteration {:d}'.format(cur_iter))

    tb_writer.close()
    logging.info('Finsh! This is train_val_{}'.format(model_name))
    logging.info(os.path.abspath(experiment_dir))

class NegativeL1Loss(nn.Module):
    def __init__(self, offset=2):
        super(NegativeL1Loss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.offset = offset

    def forward(self, input1, input2):
        loss = self.offset - self.l1_loss(input1, input2)
        # 如果loss小于0报错
        if loss < 0:
            raise ValueError('Negative loss')
        return loss