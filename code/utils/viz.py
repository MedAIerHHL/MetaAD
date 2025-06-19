import torch
import os

import cv2
import numpy as np
from SimpleITK import Image
from matplotlib import pyplot as plt
from torchvision.utils import make_grid, save_image

def create_bwr_colormap():
    colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        if i < 128:
            colormap[i] = [255, i * 2, i * 2]  # 从红色到白色
        else:
            colormap[i] = [255 - (i - 128) * 2, 255 - (i - 128) * 2, 255]  # 从白色到蓝色
    return colormap

def show_image(imgs, img_name, save_path, denormalize=True, grid_nrow=8, colormap = 'hot'):
    # Create save path
    save_path = os.path.join(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, img_name)

    # Denormalization [-1, 1] --> [0, 1]
    if isinstance(imgs, list):
        grid_nrow = imgs[0].shape[0]
        imgs = torch.cat(imgs, dim=0)

    if denormalize:
        out_imgs = torch.clamp((imgs + 1) / 2, min=0.0, max=1.0)
    else:
        out_imgs = torch.clamp(imgs, min=0.0, max=1.0)

    # Save images
    grid = make_grid(out_imgs, nrow=grid_nrow)
    save_image(grid, filename)

    # Apply color map
    im_data = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if colormap == 'bwr':
        color_image = cv2.applyColorMap(im_data, create_bwr_colormap())
    else:
        color_image = cv2.applyColorMap(im_data, cv2.COLORMAP_HOT)
    
    cv2.imwrite(filename, color_image)

def show_save_img(imgs, img_name, save_path, vmin=-2, vmax=2, colormap = 'bwr'):
    # Create save path
    save_path = os.path.join(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, img_name)

    if isinstance(imgs, list):
        grid_nrow = imgs[0].shape[0]
        imgs = torch.cat(imgs, dim=0)
    imgs_np = imgs.cpu().numpy()[:,0,:,:]
    rows, columns = imgs_np.shape[0]//grid_nrow, grid_nrow
    fig, axs = plt.subplots(rows, columns, figsize=(20, 8))
    for i, ax in enumerate(axs.flat):
        img = imgs_np[i]
        ax.imshow(img, cmap=colormap, vmin=vmin, vmax=vmax, origin='upper')
        ax.axis('off')  # 不显示坐标轴
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    # plt.subplots_adjust(left=0, bottom=0, wspace=0, hspace=0)
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()