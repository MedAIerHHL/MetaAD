import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from sympy.core.random import shuffle, random
from torch.utils.data import Dataset, DataLoader

categories = {
    'NC': [1, 0, 0, 0],
    'PD': [0, 1, 0, 0],
    # 'MSA': [0, 0, 1, 0],
    # 'PSP': [0, 0, 0, 1]
}

class UnpairedDataset(Dataset):
    """
    This dataset class can load unaligned/unpaired datasets.
    """

    def __init__(
            self,
            data_dir,
            A_path_list,
            B_path_list,
            is_paired=False,
            img_size=None,
            resize_mode='interpolate', # padding
            normalize=True,
            mode='train',
            dim=3,
    ):
        self.data_dir = data_dir
        self.A_paths = A_path_list
        self.B_paths = B_path_list
        self.is_paired = is_paired
        self.img_size = img_size
        self.resize_mode = resize_mode
        self.normalize = normalize
        self.mode = mode

        self.data = []
        for path in self.A_paths:
            category = path.split('_')[0]
            label = categories[category]
            img_id =  path.split('/')[1]
            img_name = f"{category}_{img_id}"
            A_abs_path = os.path.join(data_dir, path)
            if is_paired:
                B_abs_path = os.path.join(data_dir, self.B_paths[self.A_paths.index(path)])
            else:
                B_abs_path = os.path.join(data_dir, self._get_unpaired_img_path())
            # check if the file exists
            if os.path.exists(A_abs_path) is False:
                raise ValueError(f"{A_abs_path} does not exist.")
            if B_abs_path is not None and os.path.exists(B_abs_path) is False:
                raise ValueError(f"{B_abs_path} does not exist.")
            self.data.append([A_abs_path, B_abs_path, label, img_name])
        random.shuffle(self.data)


    def __len__(self):
        return len(self.A_paths)


    def _get_unpaired_img_path(self):
        index = np.random.randint(0, len(self.B_paths))
        return self.B_paths[index]


    def _nearest_multiple(self, number, multiple):
        remainder = number % multiple
        if remainder == 0:
            return number
        else:
            return number + (multiple - remainder)


    def _load_and_resize_img(self, image_path, image_size, resize_mode='interpolate'):
        # load
        image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(image).astype(np.float32)
        originSize = image_array.shape

        if image_size is None:
            newSize = [self._nearest_multiple(originSize[0], 8),
                       self._nearest_multiple(originSize[1], 8),
                       self._nearest_multiple(originSize[2], 8)]
        else:
            newSize = image_size

        if resize_mode == 'interpolate':
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputDirection(image.GetDirection())
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetSize(newSize)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetOutputSpacing(image.GetSpacing())
            resampler.SetOutputPixelType(sitk.sitkFloat32)
            resized_img = resampler.Execute(image)
            resized_img_array = sitk.GetArrayFromImage(resized_img)
        elif resize_mode == 'padding':
            pad_value = image_array.min()
            padding = [
                (max(0, (newSize[i] - originSize[i]) // 2),
                (max(0, (newSize[i] - originSize[i]) + 1) // 2)) for i in range(len(newSize))
            ]
            resized_img_array = np.pad(image_array, (padding[0], padding[1], padding[2]), 'constant', constant_values=pad_value)
        else:
            raise ValueError('Invalid resize mode')

        return resized_img_array


    def _normalize(self, image: torch.tensor, min_percentile: float = 0, max_percentile: float = 100):
        # clip
        min_value = np.percentile(image.numpy(), np.max([0, min_percentile]))
        max_value = np.percentile(image.numpy(), np.min([100, max_percentile]))
        image = torch.clamp(image, min=min_value, max=max_value)
        # normalize
        norm_image = (image - min_value) / (max_value - min_value)  # [0, 1]
        norm_image = torch.clamp(norm_image * 2 - 1, min=-1.0, max=1.0)  # [-1 ,1]
        return norm_image


    def __getitem__(self, index):
        # load image and class label
        A_img_path, B_img_path, label, img_name = self.data[index]
        A_img = self._load_and_resize_img(A_img_path, self.img_size, self.resize_mode)
        B_img = self._load_and_resize_img(B_img_path, self.img_size, self.resize_mode)
        A_img = torch.tensor(A_img, dtype=torch.float32).unsqueeze(0)
        B_img = torch.tensor(B_img, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32)

        if self.normalize:
            A_img = self._normalize(A_img)
            B_img = self._normalize(B_img)

        return {'A': A_img, 'B': B_img, 'A_path': A_img_path, 'B_path': B_img_path, 'paired_flag': self.is_paired, 'label': label, 'img_name': str(img_name)}

    def create_iterator(self, num):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=num,
                drop_last=True,
                shuffle=True,
            )
            for item in sample_loader:
                yield item




class InfiniteSamplerWrapper(torch.utils.data.Sampler):
    """Data sampler wrapper"""

    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def InfiniteSampler(n):
    """Data sampler"""
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            order = np.random.permutation(n)
            i = 0