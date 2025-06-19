import logging

import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import sklearn.metrics
import torchmetrics.classification


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
#                 Metrics
# ----------------------------------------
def compute_psnr(image1, image2, denormalize=False, mask=None):
    # check image format
    assert isinstance(image1, torch.Tensor)
    assert isinstance(image2, torch.Tensor)
    assert image1.shape == image2.shape, "Input images must have the same shape"

    # transform data type
    if denormalize:
        image1 = torch.clamp((image1.clone() + 1.0) / 2, min=0.0, max=1.0)
        image2 = torch.clamp((image2.clone() + 1.0) / 2, min=0.0, max=1.0)
    else:
        image1 = torch.clamp(image1.clone(), min=0.0, max=1.0)
        image2 = torch.clamp(image2.clone(), min=0.0, max=1.0)
    image1 = to_ndarray(torch.squeeze(image1))
    image2 = to_ndarray(torch.squeeze(image2))

    # compute metrics
    if mask is not None:
        assert isinstance(mask, np.ndarray)
        mask_bool = mask.astype(bool)
        image_pred = image1[mask_bool]
        image_true = image2[mask_bool]
    else:
        image_pred = image1
        image_true = image2
    psnr_value = peak_signal_noise_ratio(image_true, image_pred, data_range=1.0)
    return psnr_value


def compute_ssim(image1, image2, denormalize=False, mask=None):
    # check image format
    assert isinstance(image1, torch.Tensor)
    assert isinstance(image2, torch.Tensor)
    assert image1.shape == image2.shape, "Input images must have the same shape"

    # transform data type
    if denormalize:
        image1 = torch.clamp((image1.clone() + 1.0) / 2, min=0.0, max=1.0)
        image2 = torch.clamp((image2.clone() + 1.0) / 2, min=0.0, max=1.0)
    else:
        image1 = torch.clamp(image1.clone(), min=0.0, max=1.0)
        image2 = torch.clamp(image2.clone(), min=0.0, max=1.0)
    image1 = to_ndarray(torch.squeeze(image1))
    image2 = to_ndarray(torch.squeeze(image2))

    # compute metrics
    if mask is not None:
        assert isinstance(mask, np.ndarray)
        mask_bool = mask.astype(bool)
        image_pred = image1[mask_bool]
        image_true = image2[mask_bool]
    else:
        image_pred = image1
        image_true = image2
    ssim_value = structural_similarity(image_true, image_pred, data_range=1.0)
    return ssim_value


# ----------------------------------------
#           binaryclf_performance
# ----------------------------------------
def binaryclf_performance(y_true, y_prob, threshold=0.5):
    """
    Calculate the performance of the classifer model
    模型性能计算：
        mcc: 混淆矩阵
        accuracy: 精确度
        precision: 精准度
        recall: 召回率
        f1: f1值
        f2: f2值
        f3: f3值
        sensitivity: 灵敏度
        specificity: 特异性
        roc_auc：ROC曲线面积

    Args:
        y_prob: the predicted probabilities
        y_test: the true labels
        threshold: the threshold for the predicted probabilities
    Returns:
        preformance: the performance of the classifier model - a dictionary
    """
    # Calculate the performance of the model
    # Return the performance of the model

    # y_pred = (np.array(y_prob) > threshold).astype(int)
    
    prob_f1 = [1 if s >= threshold else 0 for s in y_prob]
    acc_f1 = accuracy_score(y_true, prob_f1)
    f1 = f1_score(y_true, prob_f1)
    recall = sklearn.metrics.recall_score(y_true, prob_f1)
    precision = sklearn.metrics.precision_score(y_true, prob_f1)
    mcc = confusion_matrix(y_true, prob_f1)
    tn, fp, fn, tp = mcc.ravel()
    accuracy_ = (tn + tp) / (tn + fp + fn + tp)
    precision_ = tp / (tp + fp)
    recall_ = tp / (tp + fn)
    f1_ = 2 * tp / (2 * tp + fp + fn)
    f2 = 5 * tp / (5 * tp + 4 * fp + fn)
    f3 = 10 * tp / (10 * tp + 9 * fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    roc_auc = roc_auc_score(y_true, y_prob)
    performance = {
        'mcc': mcc,
        'acc': acc_f1,
        'pre': precision,
        'rec': recall,
        'f1': f1,
        'f2': f2,
        'f3': f3,
        'sen': recall,
        'spe': specificity,
        'roc_auc': roc_auc,
        }
    return performance

def auc_score(y_true, y_pred_proba, threshold=0.5):
    return binaryclf_performance(y_true, y_pred_proba, threshold)['roc_auc']


def spe_score(y_true, y_pred_proba, threshold=0.5):
    return binaryclf_performance(y_true, y_pred_proba, threshold)['spe']


def sen_score(y_true, y_pred_proba, threshold=0.5):
    return binaryclf_performance(y_true, y_pred_proba, threshold)['sen']


def acc_score(y_true, y_pred_proba, threshold=0.5):
    return binaryclf_performance(y_true, y_pred_proba, threshold)['acc']




def bootstrap_confidence_interval(y_true, y_pred_prob, statistic, n_iterations=1000, alpha=0.05, cutoff=None):
    # 初始统计量
    stat = statistic(y_true, y_pred_prob, cutoff)
    # 创建空数组来存储每次重复后的统计量值
    stats = np.zeros(n_iterations)
    for i in range(n_iterations):
        # 重复抽样
        idx = np.random.randint(y_true.shape[0], size=y_true.shape[0])
        y_true_resample = y_true.iloc[idx]
        y_pred_prob_resample = y_pred_prob.iloc[idx]
        if len(np.unique(y_true_resample)) == 1:
            continue
        # 计算统计量
        stat_resample = statistic(y_true_resample, y_pred_prob_resample, cutoff)
        stats[i] = stat_resample

    stat = sum(stats)/len(stats)
    # 计算置信区间
    lower = np.percentile(stats, alpha/2*100)
    upper = np.percentile(stats, (1-alpha/2)*100)

    # print("Statistic:", stat)
    # print("Confidence interval: (%.4f, %.4f)" % (lower, upper))
    return stat, lower, upper

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.metrics import f1_score, balanced_accuracy_score
from utils.tools import find_best_cutoff
def image_wise_anomaly_detection(cfg, score_dict, image_label_dict):
    # cutoff_function = find_best_cutoff(cfg)
    score = []
    label = []
    for key in score_dict.keys():
        if key in image_label_dict.keys():
            score.append(float(score_dict[key]))
            label.append(int(image_label_dict[key]))
    auc = roc_auc_score(label, score)
    fpr, tpr, thres = roc_curve(label, score)    
    sum_ = [tpr[i]-fpr[i] for i in range(len(thres))]
    idx = np.argmax(np.array(sum_))
    TT = thres[idx]
    binperf_dict = binaryclf_performance(label, score, TT)
    prob_f1 = [1 if s >= TT else 0 for s in score]
    acc_f1 = accuracy_score(label, prob_f1)
    f1 = f1_score(label, prob_f1)
    # label_pd = pd.DataFrame(label)
    # score_pd = pd.DataFrame(score)
    #
    # mode_sen, mode_sen_lower, mode_sen_upper = bootstrap_confidence_interval(label_pd, score_pd, sen_score, cutoff=TT)
    # mode_spe, mode_spe_lower, mode_spe_upper = bootstrap_confidence_interval(label_pd, score_pd, spe_score, cutoff=TT)
    # mode_acc, mode_acc_lower, mode_acc_upper = bootstrap_confidence_interval(label_pd, score_pd, acc_score, cutoff=TT)
    # mode_auc, mode_auc_lower, mode_auc_upper = bootstrap_confidence_interval(label_pd, score_pd, auc_score, cutoff=TT)
    #
    # logging.info('Sensitivity: {:.3f} [{:.3f}, {:.3f}]'.format(mode_sen, mode_sen_lower, mode_sen_upper))
    # logging.info('Specificity: {:.3f} [{:.3f}, {:.3f}]'.format(mode_spe, mode_spe_lower, mode_spe_upper))
    # logging.info('Accuracy: {:.3f} [{:.3f}, {:.3f}]'.format(mode_acc, mode_acc_lower, mode_acc_upper))
    # logging.info('ROC-AUC: {:.3f} [{:.3f}, {:.3f}]'.format(mode_auc, mode_auc_lower, mode_auc_upper))

    return binperf_dict, TT

