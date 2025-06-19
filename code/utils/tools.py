import argparse
from datetime import datetime
import logging
import math
import random
import os
import sys

import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from torch import nn as nn
from torch.optim import Adam, SGD, lr_scheduler
import torch.backends.cudnn as cudnn
from shutil import copyfile
from datetime import datetime
import cv2
import SimpleITK as sitk

# ----------------------------------------
#             Reproducibility
# ----------------------------------------
def setup_seed(seed=42):
    random.seed(seed)  # python seed
    # 设置python哈希种子，for certain hash-based operations (e.g., the item order in
    # a set or a dict）。seed为0的时候表示不用这个feature，也可以设置为整数。 有时候需要在终端执行，到脚本实行可能就迟了。
    os.environ['PYTHONHASHSEED'] = str(seed)
    # If you or any of the libraries you are using rely on NumPy,
    # 比如Sampling，或者一些augmentation。
    # 哪些是例外可以看https://pytorch.org/docs/stable/notes/randomness.html
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为当前CPU设置随机种子。 pytorch官网倒是说(both CPU and CUDA)
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 使用多块GPU时，均设置随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 设置为True时，cuDNN使用非确定性算法寻找最高效算法
    torch.backends.cudnn.enabled = True  # pytorch使用CUDANN加速，即使用GPU加速


# ----------------------------------------
#          Data Type Conversion
# ----------------------------------------

def to_ndarray(tensor):
    """Converts a `torch.Tensor` to `numpy.ndarray`."""
    assert isinstance(tensor, torch.Tensor)
    return tensor.detach().cpu().numpy()

def to_tensor(array):
    """Converts a `numpy.ndarray` to `torch.Tensor`."""
    assert isinstance(array, np.ndarray)
    return torch.from_numpy(array).type(torch.FloatTensor).cuda()



# ----------------------------------------
#             YAML Parser
# ----------------------------------------

# yaml文件读取
def yaml_parser(yaml_path):
    with open(yaml_path, 'r') as file:
        opt = argparse.Namespace(**yaml.load(file.read(), Loader=yaml.FullLoader))
    opt.GLOBAL = argparse.Namespace(**opt.GLOBAL)
    opt.TRAIN = argparse.Namespace(**opt.TRAIN)
    opt.VAL = argparse.Namespace(**opt.VAL)
    opt.TEST = argparse.Namespace(**opt.TEST)
    opt.MODEL = argparse.Namespace(**opt.MODEL)
    opt.LOSS = argparse.Namespace(**opt.LOSS)
    opt.OPTIMIZER = argparse.Namespace(**opt.OPTIMIZER)

    return opt




# ----------------------------------------
#             dir setting
# ----------------------------------------
# 初始化设置
def init_setting(cfg, args):
    timestr = str(datetime.now().strftime('%Y_%m%d_%H%M'))
    experiment_dir = Path(cfg.GLOBAL.save_result_dir)
    experiment_dir.mkdir(exist_ok=True) # 保存实验结果的总目录
    experiment_dir = experiment_dir.joinpath(cfg.GLOBAL.exp_name)
    experiment_dir.mkdir(exist_ok=True)  # 每次实验的根目录
    experiment_dir = experiment_dir.joinpath(timestr)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)    # 保存模型的目录
    tensorboard_dir = experiment_dir.joinpath('tensorboard/')
    tensorboard_dir.mkdir(exist_ok=True)    # 保存日志的目录
    setting_dir = experiment_dir.joinpath('setting/')
    setting_dir.mkdir(exist_ok=True)            # 保存scr的目录

    copyfile(args.yaml, str(setting_dir) + '/' + os.path.basename(args.yaml))

    return experiment_dir, checkpoints_dir, tensorboard_dir




# ----------------------------------------
#                 Logging
# ----------------------------------------

# open the log file
def open_log(log_path, log_name = None):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    time_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if log_name is None:
        log_name = time_name
    else:
        log_name = log_name + '_' + time_name
    if os.path.isfile(os.path.join(log_path, '{}.log'.format(log_name))):
        os.remove(os.path.join(log_path, '{}.log'.format(log_name)))
    initLogging(os.path.join(log_path, '{}.log'.format(log_name)))

# Init for logging
def initLogging(logFilename):
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s-%(levelname)s] %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S',
                        filename=logFilename,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s-%(levelname)s] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

#  logginge日志
class ConsoleLogger:
    def __init__(self, log_file):
        # 创建一个名为 console_logger 的 logger 对象
        self.console_logger = logging.getLogger("console_logger")
        self.console_logger.setLevel(logging.DEBUG)

        # 创建一个名为 console_handler 的控制台处理器对象，用于将日志输出到控制台
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        # 创建一个名为 file_handler 的文件处理器对象，用于将日志写入到文件中
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)

        # 创建一个日志格式化器对象
        formatter = logging.Formatter("%(message)s")

        # 将日志格式化器对象添加到文件处理器和控制台处理器对象中
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 将控制台处理器对象和文件处理器对象添加到 logger对象中
        self.console_logger.addHandler(file_handler)

    def __enter__(self):
        # 重定向标准输出流和标准错误流
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # 恢复标准输出流和标准错误流
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def write(self, message):
        # 将标准输出流和标准错误流的内容写入到日志中
        self.console_logger.debug(message.rstrip())

    def flush(self):
        # 确保所有日志都被写入到文件中
        for handler in self.console_logger.handlers:
            handler.flush()

# def logger(output_dir, log_file_name):
#     # 检查日志文件是否存在，如果不存在则创建
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     log_file = '{}.log'.format(log_file_name)
#     log_path = os.path.join(output_dir, log_file)
#     if not os.path.exists(log_path):
#         with open(log_path, "w"):
#             pass
#     return ConsoleLogger(log_path)

class Logger(object):
    def __init__(self, output_dir, log_file_name, stream=sys.stdout):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_file = '{}.log'.format(log_file_name)
        log_path = os.path.join(output_dir, log_file)

        self.terminal = stream
        self.log = open(log_path, 'a+')

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# def logger(result_dir, log_file_name):
#     type = sys.getfilesystemencoding()
#     # 记录正常的 print 信息
#     sys.stdout = Logger(result_dir, log_file_name, sys.stdout)
#     # 记录 traceback 异常信息
#     sys.stderr = Logger(result_dir, log_file_name, sys.stderr)




# ----------------------------------------
#                   Env
# ----------------------------------------

def get_envs():
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    rank = int(os.getenv('RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    return local_rank, rank, world_size

# GPU信息
def GPU_info(i = 0):
    logging.info('GPU is {}'.format(torch.cuda.is_available()))
    logging.info('GPU number: {}'.format(torch.cuda.device_count()))
    logging.info('GPU: {}'.format(torch.cuda.get_device_name(i)))
    torch.cuda.empty_cache()



# early stop
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True



# ----------------------------------------
#               Save Results
# ----------------------------------------

def save_model(net, net_name, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_name = os.path.join(save_path, net_name)
    net = net.module if isinstance(net, torch.nn.DataParallel) else net
    torch.save({'model': net.state_dict()}, model_name)

# def save_model(model, checkpoints_dir, cur_iter, net_name, best=False):
#     if best:
#         save_path = os.path.join(checkpoints_dir, '{}_best.pth'.format(net_name))
#     else:
#         save_path = os.path.join(checkpoints_dir, '{}_iter_{}.pth'.format(net_name, cur_iter))
#     model = model.module if isinstance(model, torch.nn.DataParallel) else model
#     torch.save({
#         'iter': cur_iter,
#         'model_state_dict': model.state_dict(),
#     }, save_path)
#     logging.info('INFO: Save model to {}'.format(save_path))

# def load_model(model, checkpoints_dir, net_name, best = False, optimizer=None, **kwargs):
#     if best:
#         load_path = os.path.join(checkpoints_dir, '{}_best.pth'.format(net_name))
#     else:
#         load_path = os.path.join(checkpoints_dir, '{}_iter_{}.pth'.format(net_name, kwargs['cur_iter']))
#     if not os.path.exists(load_path):
#         logging.info('WARNING: No checkpoint found at {}'.format(load_path))
#         return model, optimizer, 0
#     checkpoint = torch.load(load_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     cur_iter = checkpoint['iter']
#     logging.info('INFO: Load model from {}'.format(load_path))
#     if optimizer is not None:
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         logging.info('INFO: Load optimizer from {}'.format(load_path))
#     cur_iter = checkpoint['iter']
#     return model, optimizer, cur_iter

def load_model(process_net, pretrained_file):
    pretrained_dict = torch.load(pretrained_file)['model']
    process_net.load_state_dict(pretrained_dict)
    return process_net

def save_vol_image(vol_img, img_name, save_path, denormalize=True, spacing=2.0):
    # Create save path
    save_path = os.path.join(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, img_name)

    # Denormalization [-1, 1] --> [0, 1]
    if denormalize:
        vol_img = torch.clamp((vol_img.clone() + 1) / 2, min=0.0, max=1.0)
    elif denormalize == None:
        vol_img = vol_img
    else:
        vol_img = torch.clamp(vol_img.clone(), min=0.0, max=1.0)

    # Tensor --> SimpleITK Image
    vol_img_np = vol_img.squeeze().detach().cpu().numpy()
    out_img = sitk.GetImageFromArray(vol_img_np[:, ::-1, :])

    # Set spacing
    spacings = (spacing, spacing, spacing)
    out_img.SetSpacing(spacings)

    # Save image
    sitk.WriteImage(out_img, filename)

def get_error_map(pred_images, gt_images, norm_type=None):
    error_map_list = []
    for i in range(pred_images.shape[0]):
        if norm_type == "absolute":
            error_map = np.abs(gt_images[i] - pred_images[i])
            error_map_norm = (error_map - error_map.min()) / (error_map.max() - error_map.min())  # [0, 1]
            error_map_norm = torch.clamp(error_map_norm * 2 - 1, min=-1, max=1)  # [-1, 1]
        elif norm_type == "relative":
            error_map = gt_images[i] - pred_images[i]
            end_value = torch.abs(error_map).max()
            error_map_norm = (error_map + end_value) / (end_value * 2)
            error_map_norm = torch.clamp(error_map_norm * 2 - 1, min=-1, max=1)  # [-1, 1]
        elif norm_type is None:
            error_map = gt_images[i] - pred_images[i]
            error_map_norm = error_map
        error_map_list.append(error_map_norm)
    return torch.stack(error_map_list).cuda()

# ----------------------------------------
#              cutoff
# ----------------------------------------
# cutoff 选择
def find_best_pr_auc_cutoff(pred_proba, test_label):
    best_pr_auc = 0
    best_cutoff = 0
    for cutoff in np.linspace(0, 1, 100):
        pred = (pred_proba[:,1] > cutoff).astype(int)
        pr_auc = average_precision_score(test_label, pred_proba[:,1])
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_cutoff = cutoff
    return best_pr_auc, best_cutoff

def find_best_f1_score_cutoff(pred_proba, test_label):
    best_f1_score = 0
    best_cutoff = 0
    for cutoff in np.linspace(0, 1, 100):
        pred = (pred_proba[:,1] > cutoff).astype(int)
        f1 = f1_score(test_label, pred)
        if f1 > best_f1_score:
            best_f1_score = f1
            best_cutoff = cutoff
    return best_f1_score, best_cutoff

def find_best_youden_index_cutoff(pred_proba, test_label):
    best_youden_index = 0
    best_cutoff = 0
    fpr, tpr, thres = roc_curve(test_label, pred_proba)
    sum_ = [tpr[i] - fpr[i] for i in range(len(thres))]
    idx = np.argmax(np.array(sum_))
    best_youden_index = idx
    best_cutoff = thres[idx]
    return best_youden_index, best_cutoff

def find_default_cutoff(pred_proba, test_label):
    return None, 0.5

def find_best_cutoff(cfg):
    if cfg.GLOBAL.cutoff_way == 'pr-auc':
        cutoff_function = find_best_pr_auc_cutoff
    elif cfg.GLOBAL.cutoff_way == 'youden-index':
        cutoff_function = find_best_youden_index_cutoff
    elif cfg.GLOBAL.cutoff_way == 'f1-score':
        cutoff_function = find_best_f1_score_cutoff
    elif cfg.GLOBAL.cutoff_way == 'default':
        cutoff_function = find_default_cutoff
    else:
        cutoff_function = None
        print('Warning: unknown cutoff function.')
        sys.exit(0)
    return cutoff_function



# 模型参数
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info = f'Total : {str(total_num / 1000 ** 2)} M, Trainable: {str(trainable_num / 1000 ** 2)} M'
    return info


def read_split_data(cfg, mode):
    # 遍历文件夹，一个文件夹对应一个类别
    classes = [cla for cla in os.listdir(cfg.GLOBAL.TRAIN_DIR)]
    # 排序，保证顺序一致
    classes.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(classes))

    images_path = []
    images_label = []
    every_class_num = []
    
    if mode == 'train':
        for cla in classes:
            train_cla_path = os.path.join(cfg.GLOBAL.TRAIN_DIR, cla)
            images = [os.path.join(cfg.GLOBAL.TRAIN_DIR, cla, i) for i in os.listdir(train_cla_path)]
            image_class = class_indices[cla]
            every_class_num.append(len(images)) # 记录每个类别下的图片个数
            for img_path in images:
                images_path.append(img_path)
                images_label.append(image_class)
    elif mode == 'val':
        for cla in classes:
            val_cal_path = os.path.join(cfg.GLOBAL.VAL_DIR, cla)
            # 遍历获取supported支持的所有文件路径
            images = [os.path.join(cfg.GLOBAL.VAL_DIR, cla, i) for i in os.listdir(val_cal_path)]
            # 获取该类别对应的索引
            image_class = class_indices[cla]
            # 记录该类别的样本数量
            every_class_num.append(len(images))
            for img_path in images:
                images_path.append(img_path)
                images_label.append(image_class)

    return every_class_num, images_path, images_label


def plot_image(num_classes, every_class_num, experiment_dir, mode):

    plt.bar(range(len(num_classes)), every_class_num, align='center')
    # 将横坐标0,1,2,3,4替换为相应的类别名称
    plt.xticks(range(len(num_classes)), num_classes)
    # 在柱状图上添加数值标签
    for i, v in enumerate(every_class_num):
        plt.text(x=i, y=v + 5, s=str(v), ha='center')
    # 设置x坐标
    if mode == 'train':
        plt.xlabel('train image class')
    elif mode == 'val':
        plt.xlabel('val image class')
    # 设置y坐标
    plt.ylabel('number of images')
    # 设置柱状图的标题
    plt.title('class distribution')
    if mode == 'train':
        plt.savefig(os.path.join(experiment_dir, 'train_dataset.png'))
        plt.close()
    elif mode == 'val':
        plt.savefig(os.path.join(experiment_dir, 'val_dataset.png'))
        plt.close()


def build_scheduler(optimizer, cfg):
    epochs = cfg.GLOBAL.EPOCH_NUM
    if cfg.OPTIMIZER.LR_NAME == 'linear_lr':
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - cfg.OPTIMIZER.LR_LRF) + cfg.OPTIMIZER.LR_LRF 
    elif cfg.OPTIMIZER.LR_NAME == 'cosine_lr':
        lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (1.0 - cfg.OPTIMIZER.LR_LRF) + cfg.OPTIMIZER.LR_LRF

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    return scheduler


def build_optimizer(model, cfg, logger):
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if cfg.OPTIMIZER.NAME == "Adam":
        optimizer = Adam(g0, lr=cfg.OPTIMIZER.LEARNING_RATE, 
                               betas=[cfg.OPTIMIZER.BETA1, cfg.OPTIMIZER.BETA2])
        optimizer.add_param_group({'params': g1, 'weight_decay': cfg.OPTIMIZER.WEIGHT_DECAY})  # add g1 with weight_decay
        optimizer.add_param_group({'params': g2})  # add g2 (biases)
    elif cfg.OPTIMIZER.NAME == "SGD":
        optimizer = SGD(g0, lr=cfg.OPTIMIZER.LEARNING_RATE, 
                        momentum=cfg.OPTIMIZER.MOMENTUM, 
                        nesterov=cfg.OPTIMIZER.NESTEROV)
        optimizer.add_param_group({'params': g1, 'weight_decay': cfg.OPTIMIZER.WEIGHT_DECAY})  # add g1 with weight_decay
        optimizer.add_param_group({'params': g2})  # add g2 (biases)
    logger.info(f"{'optimizer:'} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
    del g0, g1, g2

    return optimizer


def weights_init(net, init_type: str='xavier', init_gain: float=0.02):
    """ Initialize network weights.
    :param net: is the network to be initialized
    :param init_type: is the name of an initialization method [normal | xavier | kaiming | orthogonal]
    :param init_gain: is scaling factor for [normal | xavier | orthogonal].
    """
    def init_func(m):
        classname = m.__class__.__name__

        if classname.find('Norm3d') != -1:
            # Weight
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, init_gain)
            # Bias
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif hasattr(m, 'weight') and classname.find('Conv') != -1 or classname.find('Linear') != -1:
            # Weight
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                m.reset_parameters()
            # Bias
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    # Apply the initialization function
    net.apply(init_func)
