import os
import re
import random
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np



# One-hot class labels
categories = {
    'NC': [1, 0, 0, 0],
    'PD': [0, 1, 0, 0],
    'MSA': [0, 0, 1, 0],
    'PSP': [0, 0, 0, 1]
}

class PETDataset(Dataset):
    def __init__(self, data_root, data_list, image_size=None, normalize=False):
        super().__init__()
        self.image_size = image_size
        self.normalize = normalize

        # collect readable data
        self.data = []
        with open(os.path.join(data_root, data_list), "r") as file:
            for line in file:
                data_name = line.strip()
                # Get patient ID
                # patient_id = extract_patient_id(data_name)
                patient_id = data_name.split('/')[1]
                # Get class label
                category = data_name.split('_')[0]
                label = categories[category]
                # Get image paths
                if "_unpaired_" in data_name:
                    FDG_img_path = os.path.join(data_root, data_name).replace('\\', '/')
                    CFT_img_path = None
                elif "_paired_" in data_name:
                    FDG_img_path = os.path.join(data_root, data_name, f"{patient_id}_FDG.img").replace('\\', '/')
                    CFT_img_path = os.path.join(data_root, data_name, f"{patient_id}_CFT.img").replace('\\', '/')
                # check if the file exists
                if os.path.exists(FDG_img_path) is False:
                    raise ValueError(f"{FDG_img_path} does not exist.")
                if CFT_img_path is not None and os.path.exists(CFT_img_path) is False:
                    raise ValueError(f"{CFT_img_path} does not exist.")
                # Target image name
                if FDG_img_path.split('/')[-2] == "baseline":
                    input_img_name = f"{category}_{patient_id}_baseline.nii.gz"
                elif FDG_img_path.split('/')[-2] == "followup":
                    input_img_name = f"{category}_{patient_id}_followup.nii.gz"
                else:
                    input_img_name = f"{category}_{patient_id}.nii.gz"
                self.data.append([FDG_img_path, CFT_img_path, label, input_img_name])
        random.shuffle(self.data)

        # collect unpaired NC CFT data
        self.unpaired_NC_CFT_data = []
        with open(os.path.join(data_root, "NC_unpaired_DAT.txt"), "r") as file:
            for line in file:
                CFT_img_path = os.path.join(data_root, line.strip()).replace('\\', '/')
                self.unpaired_NC_CFT_data.append(CFT_img_path)
                if os.path.exists(CFT_img_path) is False:
                    raise ValueError(f"{CFT_img_path} does not exist.")
        random.shuffle(self.unpaired_NC_CFT_data)

    def __len__(self):
        return len(self.data)

    def _get_unpaired_img_path(self, class_label):
        # if class_label == categories["NC"]:
        index = np.random.randint(0, len(self.unpaired_NC_CFT_data))
        return self.unpaired_NC_CFT_data[index]

    def _normalize(self, image: torch.tensor, min_percentile: float=0, max_percentile: float=100):
        # clip
        min_value = np.percentile(image.numpy(), np.max([0, min_percentile]))
        max_value = np.percentile(image.numpy(), np.min([100, max_percentile]))
        image = torch.clamp(image, min=min_value, max=max_value)
        # normalize
        norm_image = (image - min_value) / (max_value - min_value)   # [0, 1]
        norm_image = torch.clamp(norm_image * 2 - 1, min=-1.0, max=1.0)    # [-1 ,1]
        return norm_image

    def __getitem__(self, index):
        # load image and class label
        FDG_img_path, CFT_img_path, label, input_img_name = self.data[index]
        if CFT_img_path is None:
            CFT_img_path = self._get_unpaired_img_path(label)
            is_paired = False
        else:
            is_paired = True
        FDG_img = load_and_resize_image(FDG_img_path, self.image_size)
        CFT_img = load_and_resize_image(CFT_img_path, self.image_size)

        # ndarray --> tensor
        FDG_img = torch.tensor(FDG_img, dtype=torch.float32).unsqueeze(0)
        CFT_img = torch.tensor(CFT_img, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32)

        # normalize to [-1, 1]
        if self.normalize:
            FDG_img = self._normalize(FDG_img)
            CFT_img = self._normalize(CFT_img)

        return {"FDG_img": FDG_img, "CFT_img": CFT_img, "paired_flag": is_paired,
                "label": label, "img_name": str(input_img_name)}

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )
            for item in sample_loader:
                yield item



def extract_patient_id(input_string):
    numbers = re.findall(r'\d+', input_string)
    result = ''.join(numbers)
    return result



def nearest_multiple(number, multiple):
    remainder = number % multiple
    if remainder == 0:
        return number
    else:
        return number + (multiple - remainder)



def load_and_resize_image(img_path, image_size=None):
    # load
    image = sitk.ReadImage(img_path)
    # resize
    originSize = image.GetSize()
    originSpacing = image.GetSpacing()
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)
    # resize ption 1: zero padding to multiple of 8
    if image_size is None:
        newSize = (nearest_multiple(originSize[0], 8),
                   nearest_multiple(originSize[1], 8),
                   nearest_multiple(originSize[2], 8))
        resampler.SetSize(newSize)
        resampler.SetOutputSpacing(originSpacing)
    # resize option 2: resampling (spacings changed)
    else:
        newSize = image_size
        resampler.SetSize(newSize)
        factor = originSize / np.array(newSize)
        newSpacing = originSpacing * factor
        resampler.SetOutputSpacing(newSpacing.tolist())
        resampler.SetOutputSpacing(originSpacing)
    resized_image = resampler.Execute(image)
    return (sitk.GetArrayFromImage(resized_image)).astype(float)



def InfiniteSampler(n):
    """Data sampler"""
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            # np.random.seed()
            order = np.random.permutation(n)
            i = 0



class InfiniteSamplerWrapper(torch.utils.data.Sampler):
    """Data sampler wrapper"""
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31
