import itertools
import logging
import os.path
import shutil
import time
import gc

import pandas as pd
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from torch import optim

from exp_main.loss.gan_loss import GANLoss
from exp_main.utils.tools import *
from exp_main.utils.metrics import *
from exp_main.utils.viz import *
from exp_main.data.datasets.pet_dataset_szr import PETDataset, InfiniteSamplerWrapper
# from exp_main.data.datasets.pet_dataset import UnpairedDataset, InfiniteSamplerWrapper
from exp_main.loss import get_loss
from exp_main.models.cyclegan.utils import create_generator, create_discriminator
from exp_main.src.train_val_cyclegan_loss import gen_random_mask, add_noise_to_batch


def test_each_dataset(
        cfg,
        args,
        mode,
        test_mask,
        data_dir,
        test_data_file,
        image_size,
        data_norm,
        batch_size,
        num_workers,
        vis_num,
        vis_slice,
        save_result_statue,
        netG_to_FDG,
        netG_to_CFT,
        save_test_result_dir,
        num_joints = 1,
):
    cycle_loss_list = []
    # ----------------------------------------
    #      Initialize testing parameters
    # ----------------------------------------
    # Define test dataset
    # test_A_path_lists = pd.read_csv(test_data_file)['FDG_path'].values.tolist()
    # test_B_path_lists = pd.read_csv(test_data_file)['AV45_path'].values.tolist()
    test_data = PETDataset(
        data_dir,
        test_data_file,
        image_size,
        data_norm,
    )
    logging.info('The overall number of testing images is %d' % len(test_data))

    test_loader = DataLoader(
        test_data,
        batch_size,
        num_workers,
    )
    sample_iterator_test = test_data.create_iterator(vis_num)

    test_metrics = {}
    for data_type in ["paired", "unpaired", "identity"]:
        test_metrics[f"{data_type}_psnr_FDG"] = []
        test_metrics[f"{data_type}_ssim_FDG"] = []
        test_metrics[f"{data_type}_psnr_CFT"] = []
        test_metrics[f"{data_type}_ssim_CFT"] = []

    test_losses = {}
    for data_type in ["paired", "unpaired", "identity"]:
        test_losses[f"{data_type}_loss_FDG"] = []
        test_losses[f"{data_type}_loss_CFT"] = []

    label_dict = {}
    score_dict = {}
    score_dict['|cycle_loss|'] = {}
    score_dict['|max|'] = {}
    score_dict['|min|'] = {}
    score_dict['|max-min|'] = {}
    index_dict = {}
    index_dict['|max|'] = {}
    index_dict['|min|'] = {}

    logging.info("Start testing ......")
    since = time.time()
    with torch.no_grad():
        netG_to_CFT.eval()
        netG_to_FDG.eval()
        for i, data in tqdm(enumerate(test_loader), desc="Testing"):
            if i >= 5:  # Only process the first 10 items
                break

            # Load images ([B, 1, D, H, W])
            real_FDG_img, real_CFT_img, label, img_name = data["FDG_img"].cuda().float(), data["CFT_img"].cuda().float(), data["label"].float(), data["img_name"]
            # if img_name[0] != 'PD_94970':
            #     break
            is_paired = data["paired_flag"][0]
            save_img_name = data["img_name"][0]
            category = save_img_name.split("_")[0]

            # if test_mask:
            # model_name = cfg.MODEL.name
            if False:
                if model_name == 'cyclegan_loss_noise' or model_name == 'cyclegan_loss_mask':
                    if model_name == 'cyclegan_loss_noise':
                        noise_real_FDG_img, noise_real_CFT_img = \
                            add_noise_to_batch(data["FDG_img"].float(), cfg.MODEL.noise_type, cfg.MODEL.noise_scale, cfg.MODEL.sampling_rate), \
                            add_noise_to_batch(data["CFT_img"].float(), cfg.MODEL.noise_type, cfg.MODEL.noise_scale, cfg.MODEL.sampling_rate)
                    elif model_name == 'cyclegan_loss_mask':
                        noise_real_FDG_img, noise_real_CFT_img = \
                            gen_random_mask(data["FDG_img"].float(), mask_k=cfg.MODEL.mask_k, mask_size=cfg.MODEL.mask_size,
                                            mask_prob=cfg.MODEL.mask_prob), \
                                gen_random_mask(data["CFT_img"].float(), mask_k=cfg.MODEL.mask_k,
                                                mask_size=cfg.MODEL.mask_size, mask_prob=cfg.MODEL.mask_prob)
            else:
                noise_real_FDG_img = real_FDG_img
            original_real_FDG_img = real_FDG_img.clone()
            for i in tqdm(range(100)):
                save_test_result_dir = os.path.join(save_test_result_dir, f"joint_{i}")
                if not os.path.exists(save_test_result_dir):
                    os.makedirs(save_test_result_dir)
                # if i == 0:
                #     real_FDG_img = real_FDG_img
                # else:
                #     real_FDG_img = add_noise_to_batch(real_FDG_img.float(), cfg.MODEL.noise_type,
                #                                       cfg.MODEL.noise_scale, cfg.MODEL.sampling_rate)
                fake_CFT_img = netG_to_CFT(real_FDG_img, test_mask=test_mask)
                recon_FDG_img = netG_to_FDG(fake_CFT_img, test_mask=test_mask)
                # same_FDG_img = netG_to_FDG(real_FDG_img, test_mask=test_mask)

                cycle_loss_FDG = F.l1_loss(recon_FDG_img, real_FDG_img) * cfg.LOSS.cycle_weight
                test_metrics["unpaired_psnr_FDG"].append(compute_psnr(recon_FDG_img, real_FDG_img, data_norm))
                test_metrics["unpaired_ssim_FDG"].append(compute_ssim(recon_FDG_img, real_FDG_img, data_norm))
                # test_metrics["unpaired_psnr_CFT"].append(compute_psnr(recon_CFT_img, real_CFT_img, data_norm))
                # test_metrics["unpaired_ssim_CFT"].append(compute_ssim(recon_CFT_img, real_CFT_img, data_norm))
                test_losses['unpaired_loss_FDG'].append(cycle_loss_FDG.item())
                # test_losses['unpaired_loss_CFT'].append(cycle_loss_CFT.item())

                # error map
                error_map = real_FDG_img - recon_FDG_img
                score_dict['|max|'][img_name[0]] = torch.max(error_map).item()
                try:
                    index_dict['|max|'][img_name[0]] = (error_map == torch.max(error_map).item()).nonzero().cpu()[0][2:].tolist()
                except:
                    index_dict['|max|'][img_name[0]] = [0, 0, 0]
                score_dict['|min|'][img_name[0]] = abs(torch.min(error_map).item())
                try:
                    index_dict['|min|'][img_name[0]] = (error_map == torch.min(error_map).item()).nonzero().cpu()[0][
                                                       2:].tolist()
                except:
                    index_dict['|min|'][img_name[0]] = [0, 0, 0]
                score_dict['|max-min|'][img_name[0]] = torch.max(error_map).item() - torch.min(error_map).item()
                score_dict['|cycle_loss|'][img_name[0]] = cycle_loss_FDG.item()
                label_dict[img_name[0]] = label[0][1].item()

                recon_FDG_error_map = get_error_map(recon_FDG_img, real_FDG_img, norm_type=None)
                # same_FDG_error_map = get_error_map(same_FDG_img, real_FDG_img, norm_type=None)

                if save_result_statue:
                    save_vol_image(real_FDG_img, save_img_name, os.path.join(save_test_result_dir, "input_FDG"),
                                   denormalize=None)
                    # save_vol_image(noise_real_FDG_img, save_img_name,
                    #                os.path.join(save_test_result_dir, "noise_real_FDG_img"), denormalize=None)
                    save_vol_image(fake_CFT_img, save_img_name, os.path.join(save_test_result_dir, "syn_CFT"),
                                   denormalize=None)
                    save_vol_image(recon_FDG_img, save_img_name, os.path.join(save_test_result_dir, "recon_FDG_img"),
                                   denormalize=None)
                    # save_vol_image(same_FDG_img, save_img_name, os.path.join(save_test_result_dir, "same_FDG_img"),
                    #                denormalize=None)
                    save_vol_image(recon_FDG_error_map, save_img_name,
                                   os.path.join(save_test_result_dir, "recon_FDG_error_map"), denormalize=None)
                    # save_vol_image(same_FDG_error_map, save_img_name,
                    #                os.path.join(save_test_result_dir, "same_FDG_error_map"), denormalize=None)

                # 1. for paired data, evaluate synthetic CFT image
                if is_paired:
                    # inference
                    syn_CFT_error_map = get_error_map(fake_CFT_img, real_CFT_img, norm_type=None)
                    # save results
                    if save_result_statue:
                        save_vol_image(real_CFT_img, save_img_name, os.path.join(save_test_result_dir, "real_CFT"),
                                       denormalize=None)
                        save_vol_image(syn_CFT_error_map, save_img_name,
                                       os.path.join(save_test_result_dir, "syn_CFT_error_map"), denormalize=None)
                # 归一化绝对值后的recon_FDG_error_map到[0,1]区间
                # recon_FDG_error_map = torch.abs(recon_FDG_error_map)
                # recon_FDG_error_map = (recon_FDG_error_map - torch.min(recon_FDG_error_map)) / (
                #         torch.max(recon_FDG_error_map) - torch.min(recon_FDG_error_map))
                # real_FDG_img = real_FDG_img*(1-recon_FDG_error_map) +  recon_FDG_error_map*recon_FDG_img  # calculate metrics
                # real_FDG_img = real_FDG_img*recon_FDG_error_map +  (1-recon_FDG_error_map)*recon_FDG_img
                real_FDG_img = real_FDG_img*0.9 +  0.1*recon_FDG_img
                save_test_result_dir = os.path.dirname(save_test_result_dir)

        # # 3. Visualization (FDG -> CFT -> FDG)
        # # (1) input
        # data = next(sample_iterator_test)
        # real_FDG_img = data["A_img"].cuda().float()
        # real_CFT_img = data["B_img"].cuda().float()
        # for i in data['img_name']:
        #     logging.info(i)
        # # (2) output
        # with torch.no_grad():
        #     fake_CFT_img = netG_to_CFT(real_FDG_img, test_mask=test_mask)
        #     recon_FDG_img = netG_to_FDG(fake_CFT_img, test_mask=test_mask)
        #     same_FDG_img = netG_to_FDG(real_FDG_img, test_mask=test_mask)
        # error_map_recon = get_error_map(recon_FDG_img, real_FDG_img, norm_type=None)
        # error_map_same = get_error_map(same_FDG_img, real_FDG_img, norm_type=None)
        # # (3) save
        # img_list = [torch.flip(real_FDG_img[:, :, vis_slice - 1], dims=[2]),
        #             torch.flip(fake_CFT_img[:, :, vis_slice - 1], dims=[2]),
        #             torch.flip(real_CFT_img[:, :, vis_slice - 1], dims=[2]),
        #             torch.flip(recon_FDG_img[:, :, vis_slice - 1], dims=[2]),
        #             torch.flip(same_FDG_img[:, :, vis_slice - 1], dims=[2]), ]
        # filename = "iteration_FDG_{}.png".format(mode)
        # show_save_img(img_list, filename, os.path.join(save_test_result_dir, 'samples_viz'), vmin=-1, vmax=1, colormap='hot')
        #
        # error_list = [torch.flip(error_map_recon[:, :, vis_slice - 1], dims=[2]),
        #             torch.flip(error_map_same[:, :, vis_slice - 1], dims=[2])]
        # error_filename = "iteration_error_{}.png".format(mode)
        # show_save_img(error_list, error_filename, os.path.join(save_test_result_dir, 'samples_viz'), vmin=-1, vmax=1, colormap='bwr')

        return test_metrics, test_losses, score_dict, label_dict, index_dict

def test_cyclegan(cfg, args):
    experiment_dir = os.path.join('./', cfg.TEST.test_experiment_dir)
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    test_result_dir = os.path.join(experiment_dir, 'test_result')
    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)
    test_timestr = str(datetime.now().strftime('%Y_%m%d_%H%M'))
    iter_test = cfg.TEST.iter_test
    if cfg.TEST.save_result_dir_name is not None:
        save_test_result_dir = os.path.join(test_result_dir, cfg.TEST.save_result_dir_name)
        if not os.path.exists(save_test_result_dir):
            os.makedirs(save_test_result_dir)
    else:
        save_test_result_dir = test_result_dir
    save_test_result_dir = os.path.join(save_test_result_dir, f"iter_{iter_test}_{test_timestr}")
    if not os.path.exists(save_test_result_dir):
        os.makedirs(save_test_result_dir)
    copyfile(args.yaml, str(save_test_result_dir) + '/' + os.path.basename(args.yaml))
    tensorboard_dir = os.path.join(save_test_result_dir, 'tensorboard')
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tb_writer = SummaryWriter(tensorboard_dir)

    # Loggings
    open_log(save_test_result_dir, log_name='test')
    logging.info(os.path.abspath(save_test_result_dir))
    logging.info(cfg)
    logging.info("Start testing ......")
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

    # Build generators
    if args.mode == 'test' and iter_test:
        netG_name1 = os.path.join(checkpoints_dir, 'gen_to_CFT_{:d}.pth'.format(iter_test))
        netG_name2 = os.path.join(checkpoints_dir, 'gen_to_FDG_{:d}.pth'.format(iter_test))
    else:
        netG_name1 = None
        netG_name2 = None
    netG_to_CFT = create_generator(cfg, netG_name1).cuda()
    netG_to_FDG = create_generator(cfg, netG_name2).cuda()
    
    test_data_file_dict = cfg.TEST.test_data_file_dict
    test_scores_dict_dict = {}
    test_label_dict_dict = {}
    test_index_dict_dict = {}
    for key, value in test_data_file_dict.items():
        save_test_result_dir_test = os.path.join(save_test_result_dir, key)
        if not os.path.exists(save_test_result_dir_test):
            os.makedirs(save_test_result_dir_test)
        test_metrics, test_losses, test_scores_dict, test_label_dict, test_index_dict = test_each_dataset(
            cfg,
            args,
            mode=key[:],
            test_mask=cfg.TEST.mask,
            data_dir=cfg.GLOBAL.data_dir,
            test_data_file=value,
            image_size=cfg.TEST.image_size,
            data_norm=cfg.TEST.data_norm,
            batch_size=cfg.TEST.batch_size,
            num_workers=cfg.TEST.num_workers,
            vis_num=cfg.TEST.vis_num,
            vis_slice=cfg.TEST.vis_slice,
            save_result_statue=cfg.TEST.save_result_statue,
            netG_to_FDG=netG_to_FDG,
            netG_to_CFT=netG_to_CFT,
            save_test_result_dir=save_test_result_dir_test,
            num_joints = 2,
        )
        test_scores_dict_dict[key] = test_scores_dict
        test_label_dict_dict[key] = test_label_dict
        test_index_dict_dict[key] = test_index_dict
    
    fig_hist, ax_hist = plt.subplots(2, 2)
    ax_hist_cycle = ax_hist[0, 0]
    ax_hist_abs_max = ax_hist[0, 1]
    ax_hist_abs_min = ax_hist[1, 0]
    ax_hist_abs_max_min = ax_hist[1, 1]
    # score_dict['|cycle_loss|'] = {}
    # score_dict['|max|'] = {}
    # score_dict['|min|'] = {}
    # score_dict['|max-min|'] = {}
    filename_dir = os.path.join(save_test_result_dir, 'hist')
    if not os.path.exists(filename_dir):
        os.makedirs(filename_dir)
    for score_type in next(iter(test_scores_dict_dict.values())).keys():
        if score_type == '|cycle_loss|':
            for key, value in test_scores_dict_dict.items():
                ax_hist_cycle.hist(list(value[score_type].values()), bins=20, label=score_type + '_' + key, alpha=0.5, density=True)
            # ax_hist_cycle.hist(list(train_scores_dict[score_type].values()), bins=20, label=score_type + '_train', alpha=0.5, density=True)
            # ax_hist_cycle.hist(list(val_scores_dict[score_type].values()), bins=20, label=score_type + '_val', alpha=0.5, density=True)
            # ax_hist_cycle.hist(list(test_scores_dict[score_type].values()), bins=20, label=score_type + '_test', alpha=0.5, density=True)
            # ax_hist_cycle.hist(list(test2_scores_dict[score_type].values()), bins=20, label=score_type + '_test2', alpha=0.5, density=True)
            # ax_hist_cycle.hist(list(test3_scores_dict[score_type].values()), bins=20, label=score_type + '_test3', alpha=0.5, density=True)
        elif score_type == '|max|':
            for key, value in test_scores_dict_dict.items():
                ax_hist_abs_max.hist(list(value[score_type].values()), bins=20, label=score_type + '_' + key, alpha=0.5, density=True)
            # ax_hist_abs_max.hist(list(train_scores_dict[score_type].values()), bins=20, label=score_type + '_train', alpha=0.5, density=True)
            # ax_hist_abs_max.hist(list(val_scores_dict[score_type].values()), bins=20, label=score_type + '_val', alpha=0.5, density=True)
            # ax_hist_abs_max.hist(list(test_scores_dict[score_type].values()), bins=20, label=score_type + '_test', alpha=0.5, density=True)
            # ax_hist_abs_max.hist(list(test2_scores_dict[score_type].values()), bins=20, label=score_type + '_test2', alpha=0.5, density=True)
            # ax_hist_abs_max.hist(list(test3_scores_dict[score_type].values()), bins=20, label=score_type + '_test3', alpha=0.5, density=True)
        elif score_type == '|min|':
            for key, value in test_scores_dict_dict.items():
                ax_hist_abs_min.hist(list(value[score_type].values()), bins=20, label=score_type + '_' + key, alpha=0.5, density=True)
            # ax_hist_abs_min.hist(list(train_scores_dict[score_type].values()), bins=20, label=score_type + '_train', alpha=0.5, density=True)
            # ax_hist_abs_min.hist(list(val_scores_dict[score_type].values()), bins=20, label=score_type + '_val', alpha=0.5, density=True)
            # ax_hist_abs_min.hist(list(test_scores_dict[score_type].values()), bins=20, label=score_type + '_test',alpha=0.5, density=True)
            # ax_hist_abs_min.hist(list(test2_scores_dict[score_type].values()), bins=20, label=score_type + '_test2',alpha=0.5, density=True)
            # ax_hist_abs_min.hist(list(test3_scores_dict[score_type].values()), bins=20, label=score_type + '_test3',alpha=0.5, density=True)
        elif score_type == '|max-min|':
            for key, value in test_scores_dict_dict.items():
                ax_hist_abs_max_min.hist(list(value[score_type].values()), bins=20, label=score_type + '_' + key, alpha=0.5, density=True)
            # ax_hist_abs_max_min.hist(list(train_scores_dict[score_type].values()), bins=20, label=score_type + '_train', alpha=0.5, density=True)
            # ax_hist_abs_max_min.hist(list(val_scores_dict[score_type].values()), bins=20, label=score_type + '_val',alpha=0.5, density=True)
            # ax_hist_abs_max_min.hist(list(test_scores_dict[score_type].values()), bins=20, label=score_type + '_test',alpha=0.5, density=True)
            # ax_hist_abs_max_min.hist(list(test2_scores_dict[score_type].values()), bins=20, label=score_type + '_test2',alpha=0.5, density=True)
            # ax_hist_abs_max_min.hist(list(test3_scores_dict[score_type].values()), bins=20, label=score_type + '_test3',alpha=0.5, density=True)
    ax_hist_cycle.legend()
    ax_hist_abs_max.legend()
    ax_hist_abs_min.legend()
    ax_hist_abs_max_min.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(filename_dir, 'hist.png'), dpi=600, bbox_inches='tight')
    plt.close()

    # # all_label_dict = {**val_label_dict, **test_label_dict, **test2_label_dict, **test3_label_dict}
    # all_label_dict = {}
    # for key, value in test_label_dict_dict.items():
    #     all_label_dict = {**all_label_dict, **value}
    # all_scores_dict = {}
    # for key in next(iter(test_scores_dict_dict.values())).keys():
    #     all_scores_dict[key] = {}
    #     for key, value in test_scores_dict_dict.items():
    #         all_scores_dict[key] = {**all_scores_dict[key], **value}
    #     # all_scores_dict[key] = {**val_scores_dict[key], **test_scores_dict[key], **test2_scores_dict[key], **test3_scores_dict[key]}
    # all_index_dict = {}
    # for key in next(iter(test_scores_dict_dict.values())).keys():
    #     all_index_dict[key] = {}
    #     for key, value in test_index_dict_dict.items():
    #         all_index_dict[key] = {**all_index_dict[key], **value}
    #     # all_index_dict[key] = {**val_index_dict[key], **test_index_dict[key], **test2_index_dict[key], **test3_index_dict[key]}

    # df = pd.DataFrame()
    # df_save_dir = os.path.join(save_test_result_dir, 'uad_pred')
    # if not os.path.exists(df_save_dir):
    #     os.makedirs(df_save_dir)
    # df['label'] = pd.Series(all_label_dict)
    # for key in all_label_dict.keys():
    #     df.loc[key, 'label'] = all_label_dict[key]
    #     for key2 in all_scores_dict.keys():
    #         df.loc[key, key2 + '_score'] = all_scores_dict[key2][key]
    #     for key3 in all_index_dict.keys():
    #         string_list = list(map(str, all_index_dict[key3][key]))
    #         result = ', '.join(string_list)
    #         df.loc[key, key3 + '_index'] = result
    # df.to_csv(os.path.join(df_save_dir, 'label_scores.csv'))
