import itertools
import os.path
import shutil
import time
import gc

from copy import deepcopy

import pandas as pd
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


def val_model(
        cfg,
        mode,
        cur_iter,
        val_loader,
        val_sample_iterator,
        netG_to_CFT,
        netG_to_FDG,
        exexperiment_dir,
        cycle_weight,
        identity_weight,
        data_norm = True,
        vis_slice = 27,
        vis_num = 16,
        test_mask = False,
):
    # 2. Calculate metrics
    val_metrics = {}
    for data_type in ["paired", "unpaired", "identity"]:
        val_metrics[f"{data_type}_psnr_FDG"] = []
        val_metrics[f"{data_type}_ssim_FDG"] = []
        val_metrics[f"{data_type}_psnr_CFT"] = []
        val_metrics[f"{data_type}_ssim_CFT"] = []

    val_losses ={}
    for data_type in ["paired", "unpaired", "identity"]:
        val_losses[f"{data_type}_loss_FDG"] = []
        val_losses[f"{data_type}_loss_CFT"] = []
    
    label_dict = {}
    score_dict = {}
    score_dict['|cycle_loss|'] = {}
    score_dict['|max|'] = {}
    score_dict['|min|'] = {}
    score_dict['|max-min|'] = {}
    index_dict = {}
    index_dict['|max|'] = {}
    index_dict['|min|'] = {}
    
    with torch.no_grad():
        for _, data in enumerate(val_loader):
            # (1) Load images ([B, 1, D, H, W])
            real_FDG_img, real_CFT_img, label, img_name = data["FDG_img"].cuda().float(), data["CFT_img"].cuda().float(), data["label"].float(), data["img_name"]
            is_paired = data["paired_flag"][0]
            
            val_losses['paired_loss_FDG'] = 0
            val_losses['paired_loss_CFT'] = 0

            # (2) paired data results
            if is_paired:
                syn_FDG_img = netG_to_FDG(real_CFT_img)
                syn_CFT_img = netG_to_CFT(real_FDG_img)
                val_metrics["paired_psnr_FDG"].append(compute_psnr(syn_FDG_img, real_FDG_img, data_norm))
                val_metrics["paired_ssim_FDG"].append(compute_ssim(syn_FDG_img, real_FDG_img, data_norm))
                val_metrics["paired_psnr_CFT"].append(compute_psnr(syn_CFT_img, real_CFT_img, data_norm))
                val_metrics["paired_ssim_CFT"].append(compute_ssim(syn_CFT_img, real_CFT_img, data_norm))
            
            # FDG -> CFT -> FDG
            fake_CFT_img = netG_to_CFT(real_FDG_img, test_mask = test_mask)
            recon_FDG_img = netG_to_FDG(fake_CFT_img, test_mask = test_mask)
            cycle_loss_FDG = F.l1_loss(recon_FDG_img, real_FDG_img) * cycle_weight
            # CFT -> FDG -> CFT
            fake_FDG_img = netG_to_FDG(real_CFT_img, test_mask = test_mask)
            recon_CFT_img = netG_to_CFT(fake_FDG_img, test_mask = test_mask)
            cycle_loss_CFT = F.l1_loss(recon_CFT_img, real_CFT_img) * cycle_weight
            
            val_metrics["unpaired_psnr_FDG"].append(compute_psnr(recon_FDG_img, real_FDG_img, data_norm))
            val_metrics["unpaired_ssim_FDG"].append(compute_ssim(recon_FDG_img, real_FDG_img, data_norm))
            val_metrics["unpaired_psnr_CFT"].append(compute_psnr(recon_CFT_img, real_CFT_img, data_norm))
            val_metrics["unpaired_ssim_CFT"].append(compute_ssim(recon_CFT_img, real_CFT_img, data_norm))
            val_losses['unpaired_loss_FDG'].append(cycle_loss_FDG.item())
            val_losses['unpaired_loss_CFT'].append(cycle_loss_CFT.item())
            
            # (4) identity mapping results
            same_FDG_img = netG_to_FDG(real_FDG_img, test_mask = test_mask)
            same_CFT_img = netG_to_CFT(real_CFT_img, test_mask = test_mask)
            identity_loss_FDG = F.l1_loss(same_FDG_img, real_FDG_img) * identity_weight
            identity_loss_CFT = F.l1_loss(same_CFT_img, real_CFT_img) * identity_weight
            val_metrics["identity_psnr_FDG"].append(compute_psnr(same_FDG_img, real_FDG_img, data_norm))
            val_metrics["identity_ssim_FDG"].append(compute_ssim(same_FDG_img, real_FDG_img, data_norm))
            val_metrics["identity_psnr_CFT"].append(compute_psnr(same_CFT_img, real_CFT_img, data_norm))
            val_metrics["identity_ssim_CFT"].append(compute_ssim(same_CFT_img, real_CFT_img, data_norm))
            val_losses['identity_loss_FDG'].append(identity_loss_FDG.item())
            val_losses['identity_loss_CFT'].append(identity_loss_CFT.item())
            
            # error map
            error_map = real_FDG_img - recon_FDG_img
            score_dict['|max|'][img_name[0]] = torch.max(error_map).item()
            index_dict['|max|'][img_name[0]] = (error_map == torch.max(error_map).item()).nonzero().cpu()[0][2:].tolist()
            score_dict['|min|'][img_name[0]] = abs(torch.min(error_map).item())
            index_dict['|min|'][img_name[0]] = (error_map == torch.min(error_map).item()).nonzero().cpu()[0][2:].tolist()
            score_dict['|max-min|'][img_name[0]] = torch.max(error_map).item() - torch.min(error_map).item()
            score_dict['|cycle_loss|'][img_name[0]] = cycle_loss_FDG.item()
            label_dict[img_name[0]] = label[0][1].item()

        logging.info(
                    '[Iteration {:d} | FDG -> CFT -> FDG] Cycle Loss: {:.4f}'.format(cur_iter, np.mean(val_losses['unpaired_loss_FDG']))
                )
        logging.info(
            '[Iteration {:d} | CFT -> FDG -> CFT] Cycle Loss: {:.4f}'.format(cur_iter, np.mean(val_losses['unpaired_loss_CFT']))
        )

        # (5) Log metrics for paired data
        logging.info(
            '[Iteration {:d} - {} | Synthetic CFT] PSNR: {:.4f} ± {:.4f} | SSIM: {:.4f} ± {:.4f}'
            .format(cur_iter, mode, np.mean(val_metrics["paired_psnr_CFT"]), np.std(val_metrics["paired_psnr_CFT"]),
                    np.mean(val_metrics["paired_ssim_CFT"]), np.std(val_metrics["paired_ssim_CFT"])))
        logging.info(
            '[Iteration {:d} - {} | Synthetic FDG] PSNR: {:.4f} ± {:.4f} | SSIM: {:.4f} ± {:.4f}'
            .format(cur_iter, mode, np.mean(val_metrics["paired_psnr_FDG"]), np.std(val_metrics["paired_psnr_FDG"]),
                    np.mean(val_metrics["paired_ssim_FDG"]), np.std(val_metrics["paired_ssim_FDG"])))

        # (6) Log metrics for unpaired data
        logging.info(
            '[Iteration {:d} - {} | FDG -> CFT -> FDG] PSNR: {:.4f} ± {:.4f} | SSIM: {:.4f} ± {:.4f}'
            .format(cur_iter, mode, np.mean(val_metrics["unpaired_psnr_FDG"]), np.std(val_metrics["unpaired_psnr_FDG"]),
                    np.mean(val_metrics["unpaired_ssim_FDG"]), np.std(val_metrics["unpaired_ssim_FDG"])))
        logging.info(
            '[Iteration {:d} - {} | CFT -> FDG -> CFT] PSNR: {:.4f} ± {:.4f} | SSIM: {:.4f} ± {:.4f}'
            .format(cur_iter, mode, np.mean(val_metrics["unpaired_psnr_CFT"]), np.std(val_metrics["unpaired_psnr_CFT"]),
                    np.mean(val_metrics["unpaired_ssim_CFT"]), np.std(val_metrics["unpaired_psnr_CFT"])))

        # (7) Log metrics for identity mapping data
        logging.info(
            '[Iteration {:d} - {} | CFT -> CFT] PSNR: {:.4f} ± {:.4f} | SSIM: {:.4f} ± {:.4f}'
            .format(cur_iter, mode, np.mean(val_metrics["identity_psnr_CFT"]), np.std(val_metrics["identity_psnr_CFT"]),
                    np.mean(val_metrics["identity_ssim_CFT"]), np.std(val_metrics["identity_psnr_CFT"])))
        logging.info(
            '[Iteration {:d} - {} | FDG -> FDG] PSNR: {:.4f} ± {:.4f} | SSIM: {:.4f} ± {:.4f}\n'
            .format(cur_iter, mode, np.mean(val_metrics["identity_psnr_FDG"]), np.std(val_metrics["identity_psnr_FDG"]),
                    np.mean(val_metrics["identity_ssim_FDG"]), np.std(val_metrics["identity_ssim_FDG"])))
        
        # for key in score_dict.keys():
        #     perf_dict, thre = image_wise_anomaly_detection(cfg, score_dict[key], label_dict)
        #     # logging.info(
        #     # '  key  |   AUC   PRECISION   RECALL   ACC   F-1   Thres')
            
        #     # logging.info(
        #     #         '  %s  |  %5.3f  %5.3f  %5.3f  %5.3f  %5.3f  %5.3f'
        #     #         % (
        #     #             key,
        #     #             auc * 100, precision * 100, recall * 100, acc * 100, f1 * 100, thres,))
            
        #     logging.info(
        #         '{} Auc: {:.3f}      Acc: {:.3f}      Sen: {:.3f}      Spe: {:.3f}      F1: {:.3f}      thres: {:.3f}'.format(
        #             key[0:4],
        #             perf_dict['roc_auc'],
        #             perf_dict['acc'],
        #             perf_dict['sen'],
        #             perf_dict['spe'],
        #             perf_dict['f1'],
        #             thre,
        #         ))
        
        # 3. Visualization (FDG -> CFT -> FDG)
        # (1) input
        data = next(val_sample_iterator)
        real_FDG_img = data["FDG_img"].cuda().float()
        real_CFT_img = data["CFT_img"].cuda().float()
        # (2) output
        with torch.no_grad():
            fake_CFT_img = netG_to_CFT(real_FDG_img, test_mask = test_mask)
            recon_FDG_img = netG_to_FDG(fake_CFT_img, test_mask = test_mask)
            same_FDG_img = netG_to_FDG(real_FDG_img, test_mask = test_mask)
        error_map_recon = get_error_map(recon_FDG_img, real_FDG_img, norm_type=None)
        error_map_same = get_error_map(same_FDG_img, real_FDG_img, norm_type=None)
        error_map_recon_norm = get_error_map(recon_FDG_img, real_FDG_img, norm_type='relative')
        error_map_same_norm = get_error_map(same_FDG_img, real_FDG_img, norm_type='relative')
        # (3) save
        img_list = [torch.flip(real_FDG_img[:, :, vis_slice - 1], dims=[2]),
                    torch.flip(fake_CFT_img[:, :, vis_slice - 1], dims=[2]),
                    torch.flip(real_CFT_img[:, :, vis_slice - 1], dims=[2]),
                    torch.flip(recon_FDG_img[:, :, vis_slice - 1], dims=[2]),
                    torch.flip(same_FDG_img[:, :, vis_slice - 1], dims=[2]),]
        if cfg.TEST.mask == True:
            mask_FDG_img = netG_to_CFT.gen_random_mask(real_FDG_img, cfg.MODEL.mask_k, cfg.MODEL.mask_size, cfg.MODEL.mask_prob)
            img_list.append(torch.flip(mask_FDG_img[:, :, vis_slice - 1], dims=[2]))

        filename_norm = "iteration_{:d}_norm_FDG_{}.png".format(cur_iter, mode)
        show_image(img_list, filename_norm, os.path.join(exexperiment_dir, 'samples_viz'), data_norm, vis_num)
        filename = "iteration_{:d}_FDG_{}.png".format(cur_iter, mode)
        show_save_img(img_list, filename, os.path.join(exexperiment_dir, 'samples_viz'), vmin=-1, vmax=1,colormap='hot')

        error_list_norm = [torch.flip(error_map_recon_norm[:, :, vis_slice - 1], dims=[2]),
                           torch.flip(error_map_same_norm[:, :, vis_slice - 1], dims=[2])]
        error_filename_norm = "iteration_{:d}_norm_error_{}.png".format(cur_iter, mode)
        show_image(error_list_norm, error_filename_norm, os.path.join(exexperiment_dir, 'samples_viz'), data_norm,
                   vis_num, colormap='bwr')
        error_list = [torch.flip(error_map_recon[:, :, vis_slice - 1], dims=[2]),
                      torch.flip(error_map_same[:, :, vis_slice - 1], dims=[2])]
        error_filename = "iteration_{:d}_error_{}.png".format(cur_iter, mode)
        show_save_img(error_list, error_filename, os.path.join(exexperiment_dir, 'samples_viz'), vmin=-2, vmax=2,
                      colormap='bwr')
    
    return val_metrics, val_losses, score_dict, label_dict, index_dict


def train_cyclegan(cfg, args):
    model_name = cfg.MODEL.name
    assert cfg.MODEL.noise == False and cfg.MODEL.mask == False, "The noise and mask are not supported in CycleGAN."
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

    # Define validation dataset
    val_data = PETDataset(
        cfg.GLOBAL.data_dir,
        cfg.VAL.val_data_file,
        cfg.TEST.image_size,
        cfg.TEST.data_norm,
    )
    logging.info('The overall number of validation images is %d' % len(val_data))

    val_loader = DataLoader(
        val_data,
        batch_size=cfg.TEST.batch_size,
        num_workers=cfg.TEST.num_workers,
    )
    sample_iterator_val = val_data.create_iterator(cfg.TEST.vis_num)

    # Define test dataset
    test_data = PETDataset(
        cfg.GLOBAL.data_dir,
        cfg.TEST.test_data_file,
        cfg.TEST.image_size,
        cfg.TEST.data_norm,
    )
    logging.info('The overall number of testing images is %d' % len(test_data))

    test_loader = DataLoader(
        test_data,
        batch_size=cfg.TEST.batch_size,
        num_workers=cfg.TEST.num_workers,
    )
    sample_iterator_test = test_data.create_iterator(cfg.TEST.vis_num)

    val_test_data = PETDataset(
        cfg.GLOBAL.data_dir,
        cfg.TEST.test_data_file2,
        cfg.TEST.image_size,
        cfg.TEST.data_norm,
    )
    val_test_loader = DataLoader(
        val_test_data,
        batch_size=cfg.TEST.batch_size,
        num_workers=cfg.TEST.num_workers,
        shuffle=True
    )
    sample_iterator_val_test = val_test_data.create_iterator(cfg.TEST.vis_num)


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
    # ============================================= Train-Val =============================================
    since = time.time()
    for iter_index in tqdm(range(start_iter, cfg.TRAIN.max_iter), desc='Training'):
        cur_iter = iter_index + 1
        # ============================================= Train =============================================
        # Load images ([B, 1, D, H, W])
        data = next(train_loader)
        real_FDG_img, real_CFT_img = data["FDG_img"].cuda().float(), data["CFT_img"].cuda().float()
        paired_flag = data["paired_flag"]

        # ----------------------------------------
        #            Train Generators
        # ----------------------------------------

        # FDG --> CFT --> FDG
        fake_CFT_img = netG_to_CFT(real_FDG_img)
        recon_FDG_img = netG_to_FDG(fake_CFT_img)
        # with torch.no_grad:
        #     anormal_fake_CFT_img = netG_to_CFT(anormal_real_FDG_img)

        # CFT --> FDG --> CFT
        fake_FDG_img = netG_to_FDG(real_CFT_img)
        recon_CFT_img = netG_to_CFT(fake_FDG_img)

        # identity mapping
        same_FDG_img = netG_to_FDG(real_FDG_img)
        same_CFT_img = netG_to_CFT(real_CFT_img)

        # same_FDG_img2 = netG_to_FDG(real_FDG_img)
        # same_CFT_img2 = netG_to_CFT(real_CFT_img)

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

        # (4) compute paired reconstruction loss
        recon_loss_CFT = 0
        recon_loss_FDG = 0
        paired_counts = 0
        for batch_idx, is_paired in enumerate(paired_flag):
            if is_paired:
                recon_loss_CFT += F.l1_loss(fake_CFT_img[batch_idx], real_CFT_img[batch_idx])
                recon_loss_FDG += F.l1_loss(fake_FDG_img[batch_idx], real_FDG_img[batch_idx])
                paired_counts += 1
        if paired_counts > 0:
            recon_loss_CFT = recon_loss_CFT / paired_counts * cfg.LOSS.recon_weight
            recon_loss_FDG = recon_loss_FDG / paired_counts * cfg.LOSS.recon_weight
        recon_loss = recon_loss_CFT + recon_loss_FDG

        # backward and optimize
        total_loss_G = cycle_loss + GAN_loss_G + identity_loss + recon_loss
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
        tb_writer.add_scalar('Train/cycle_loss_FDG', cycle_loss_FDG.item()/cfg.LOSS.cycle_weight, cur_iter)
        tb_writer.add_scalar('Train/cycle_loss_CFT', cycle_loss_CFT.item()/cfg.LOSS.cycle_weight, cur_iter)
        tb_writer.add_scalar('Train/cycle_loss', cycle_loss.item()/cfg.LOSS.cycle_weight, cur_iter)
        tb_writer.add_scalar('Train/identity_loss_FDG', identity_loss_FDG.item()/cfg.LOSS.identity_weight, cur_iter)
        tb_writer.add_scalar('Train/identity_loss_CFT', identity_loss_CFT.item()/cfg.LOSS.identity_weight, cur_iter)
        tb_writer.add_scalar('Train/identity_loss', identity_loss.item()/cfg.LOSS.identity_weight, cur_iter)
        # tb_writer.add_scalar('Train/recon_loss_CFT', recon_loss_CFT.item(), cur_iter)
        # tb_writer.add_scalar('Train/recon_loss_FDG', recon_loss_FDG.item(), cur_iter)
        # tb_writer.add_scalar('Train/recon_loss', recon_loss.item(), cur_iter)
        tb_writer.add_scalar('Train/total_loss_G', total_loss_G.item(), cur_iter)
        
        if cfg.TRAIN.log_interval and cur_iter % cfg.TRAIN.log_interval == 0:
            # print training status
            logging.info(
                '[Iteration {:d}] CFT Identity Loss: {:.4f} | FDG Identity Loss: {:.4f}'
                .format(cur_iter, identity_loss_CFT.item(), identity_loss_FDG.item())
            )

            if paired_counts > 0:
                logging.info(
                    '[Iteration {:d}] CFT Reconstruction Loss: {:.4f} | FDG Reconstruction Loss: {:.4f}'
                    .format(cur_iter, recon_loss_CFT.item(), recon_loss_FDG.item())
                )

            if cfg.TRAIN.gan_start_iter and cur_iter >= cfg.TRAIN.gan_start_iter:
                with torch.no_grad():
                    # (1) FDG --> CFT --> FDG
                    # real_CFT_out = netD_CFT(real_CFT_img)
                    # fake_CFT_out = netD_CFT(fake_CFT_img)
                    real_CFT_score = real_CFT_out.mean().item()
                    fake_CFT_score = fake_CFT_out.mean().item()
                    # (2) CFT --> FDG --> CFT
                    # real_FDG_out = netD_FDG(real_FDG_img)
                    # fake_FDG_out = netD_FDG(fake_FDG_img)
                    real_FDG_score = real_FDG_out.mean().item()
                    fake_FDG_score = fake_FDG_out.mean().item()
                # print training status
                logging.info(
                    '[Iteration {:d} | FDG -> CFT -> FDG] Cycle Loss: {:.4f} | D(y) / D(G(x)): {:.4f} / {:.4f}'
                    .format(cur_iter, cycle_loss_FDG.item(), real_CFT_score, fake_CFT_score)
                )
                logging.info(
                    '[Iteration {:d} | CFT -> FDG -> CFT] Cycle Loss: {:.4f} | D(x) / D(G(y)): {:.4f} / {:.4f}'
                    .format(cur_iter, cycle_loss_CFT.item(), real_FDG_score, fake_FDG_score)
                )

            else:
                # print training status
                logging.info(
                    '[Iteration {:d} | FDG -> CFT -> FDG] Cycle Loss: {:.4f}'.format(cur_iter, cycle_loss_FDG.item())
                )
                logging.info(
                    '[Iteration {:d} | CFT -> FDG -> CFT] Cycle Loss: {:.4f}'.format(cur_iter, cycle_loss_CFT.item())
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
            netG_to_FDG.eval()
            netG_to_CFT.eval()
            netD_FDG.eval()
            netD_CFT.eval()

            val_metrics, val_losses, val_scores_dict, val_label_dict, val_index_dict = val_model(
                cfg,
                'val',
                cur_iter,
                val_loader,
                sample_iterator_val,
                netG_to_CFT,
                netG_to_FDG,
                experiment_dir,
                cfg.LOSS.cycle_weight,
                cfg.LOSS.identity_weight,
                cfg.TEST.data_norm,
                cfg.TEST.vis_slice,
                cfg.TEST.vis_num,
                test_mask=cfg.TEST.mask
            )
            
            test_metrics, test_losses, test_scores_dict, test_label_dict, test_index_dict = val_model(
                cfg,
                'test',
                cur_iter,
                test_loader,
                sample_iterator_test,
                netG_to_CFT,
                netG_to_FDG,
                experiment_dir,
                cfg.LOSS.cycle_weight,
                cfg.LOSS.identity_weight,
                cfg.TEST.data_norm,
                cfg.TEST.vis_slice,
                cfg.TEST.vis_num,
                test_mask=cfg.TEST.mask
            )

            fig_hist, ax_hist = plt.subplots(2, 2)
            ax_hist_cycle = ax_hist[0, 0]
            ax_hist_abs_max = ax_hist[0, 1]
            ax_hist_abs_min = ax_hist[1, 0]
            ax_hist_abs_max_min = ax_hist[1, 1]
            # score_dict['|cycle_loss|'] = {}
            # score_dict['|max|'] = {}
            # score_dict['|min|'] = {}
            # score_dict['|max-min|'] = {}
            filename_dir = os.path.join(experiment_dir, 'hist')
            if not os.path.exists(filename_dir):
                os.makedirs(filename_dir)
            for score_type in val_scores_dict.keys():
                if score_type == '|cycle_loss|':
                    ax_hist_cycle.hist(list(val_scores_dict[score_type].values()), bins=20, label=score_type + '_val',
                                       alpha=0.5)
                    ax_hist_cycle.hist(list(test_scores_dict[score_type].values()), bins=20, label=score_type + '_test',
                                       alpha=0.5)
                elif score_type == '|max|':
                    ax_hist_abs_max.hist(list(val_scores_dict[score_type].values()), bins=20, label=score_type + '_val',
                                         alpha=0.5)
                    ax_hist_abs_max.hist(list(test_scores_dict[score_type].values()), bins=20,
                                         label=score_type + '_test', alpha=0.5)
                elif score_type == '|min|':
                    ax_hist_abs_min.hist(list(val_scores_dict[score_type].values()), bins=20, label=score_type + '_val',
                                         alpha=0.5)
                    ax_hist_abs_min.hist(list(test_scores_dict[score_type].values()), bins=20,
                                         label=score_type + '_test', alpha=0.5)
                elif score_type == '|max-min|':
                    ax_hist_abs_max_min.hist(list(val_scores_dict[score_type].values()), bins=20,
                                             label=score_type + '_val', alpha=0.5)
                    ax_hist_abs_max_min.hist(list(test_scores_dict[score_type].values()), bins=20,
                                             label=score_type + '_test', alpha=0.5)
            ax_hist_cycle.legend()
            ax_hist_abs_max.legend()
            ax_hist_abs_min.legend()
            ax_hist_abs_max_min.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(filename_dir, 'hist_{}.png'.format(cur_iter)), dpi=600, bbox_inches='tight')
            plt.close()

            val_test_scores_dict = {}
            val_test_label_dict = {**val_label_dict, **test_label_dict}
            for key in val_scores_dict.keys():
                val_test_scores_dict[key] = {}
                val_test_scores_dict[key] = {**val_scores_dict[key], **test_scores_dict[key]}
            val_test_index_dict = {}
            for key in val_index_dict.keys():
                val_test_index_dict[key] = {}
                val_test_index_dict[key] = {**val_index_dict[key], **test_index_dict[key]}

            df = pd.DataFrame()
            df_save_dir = os.path.join(experiment_dir, 'uad_pred')
            if not os.path.exists(df_save_dir):
                os.makedirs(df_save_dir)
            df['label'] = pd.Series(val_test_label_dict)
            for key in val_test_label_dict.keys():
                df.loc[key, 'label'] = val_test_label_dict[key]
                for key2 in val_test_scores_dict.keys():
                    df.loc[key, key2 + '_score'] = val_test_scores_dict[key2][key]
                for key3 in val_test_index_dict.keys():
                    string_list = list(map(str, val_test_index_dict[key3][key]))
                    result = ', '.join(string_list)
                    df.loc[key, key3 + '_index'] = result
            df.to_csv(os.path.join(df_save_dir, 'label_scores_{}.csv'.format(cur_iter)))

            for key in val_test_scores_dict.keys():
                perf_dict, thre = image_wise_anomaly_detection(cfg, val_test_scores_dict[key], val_test_label_dict)
                a = 'perf_' + key
                tb_writer.add_scalars(
                    f'Val-Test/{a}',{
                        'roc_auc': perf_dict['roc_auc'],
                        'acc': perf_dict['acc'],
                        'sen': perf_dict['sen'],
                        'spe': perf_dict['spe'],
                        'f1': perf_dict['f1'],
                        'thre': thre,
                        },
                    cur_iter)
                logging.info(
                    '{}  Auc: {:.3f}  Acc: {:.3f}  Sen: {:.3f}  Spe: {:.3f}  F1: {:.3f}  thres: {:.3f}'
                    .format(
                        key[0:6],
                        perf_dict['roc_auc'],
                        perf_dict['acc'],
                        perf_dict['sen'],
                        perf_dict['spe'],
                        perf_dict['f1'],
                        thre,
                    )
                )
            
            
            # val_test_metrics, val_test_losses = val_model(
            #     cfg,
            #     'val_test',
            #     cur_iter,
            #     val_test_loader,
            #     sample_iterator_val_test,
            #     netG_to_CFT,
            #     netG_to_FDG,
            #     experiment_dir,
            #     cfg.LOSS.cycle_weight,
            #     cfg.LOSS.identity_weight,
            #     cfg.TEST.data_norm,
            #     cfg.TEST.vis_slice,
            #     cfg.TEST.vis_num,
            #     test_mask=cfg.TEST.mask
            # )
            
            netG_to_FDG.train()
            netG_to_CFT.train()
            netD_FDG.train()
            netD_CFT.train()
            
            # 同时遍历val和test的metrics和losses把相同的key的值都保存到tb_writer中的同一张图上
            for key in val_metrics.keys():
                tb_writer.add_scalars(
                    f'Val-Test/{key}',
                    {'val': np.mean(val_metrics[key]),
                                'test': np.mean(test_metrics[key])},
                    cur_iter)

            for key in val_losses.keys():
                tb_writer.add_scalars(
                    f'Val-Test/{key}',
                    {'val': np.mean(val_losses[key]),
                                'test': np.mean(test_losses[key])},
                    cur_iter)

        # ============================================= Save =============================================
        # ----------------------------------------
        #            save model
        # ----------------------------------------
        # if (cur_iter % cfg.TRAIN.save_interval == 0 and cur_iter >= cfg.TRAIN.save_interval) or cur_iter == cfg.TRAIN.max_iter:
        #     save_model(vae, checkpoints_dir, cur_iter, "vae", best=False)

        # early stop and save model according to the train loss
        max_train_loss = cfg.TRAIN.max_train_loss  # Set the maximum train loss threshold
        if cfg.TRAIN.early_stop and total_loss_G < max_train_loss:
            early_stopping(total_loss_G)
        if early_stopping.early_stop:
            # save model
            save_model(netG_to_CFT, 'gen_to_CFT_{}.pth'.format('best'), checkpoints_dir)
            save_model(netG_to_FDG, 'gen_to_FDG_{}.pth'.format('best'), checkpoints_dir)
            save_model(netD_CFT, 'dis_to_CFT_{}.pth'.format('best'), checkpoints_dir)
            save_model(netD_FDG, 'dis_to_FDG_{}.pth'.format('best'), checkpoints_dir)
            break

    tb_writer.close()
    logging.info('Finsh! This is train_val_{}'.format(model_name))
    logging.info(os.path.abspath(experiment_dir))