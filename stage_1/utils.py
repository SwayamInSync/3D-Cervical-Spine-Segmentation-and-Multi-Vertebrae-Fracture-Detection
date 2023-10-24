import monai.transforms as transforms
import pydicom
import cv2 as cv
from glob import glob
import os
import numpy as np
import nibabel as nib
import torch
from monai.transforms import Resize
from timm.models.layers import Conv2dSame

from .conv_3d_same import Conv3dSame
from .config import Config

# train and valid data transforms

transforms_train = transforms.Compose([
    transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
    transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
    transforms.RandAffined(keys=["image", "mask"], translate_range=[int(
        x*y) for x, y in zip(Config.image_sizes, [0.3, 0.3, 0.3])], padding_mode='zeros', prob=0.7),
    transforms.RandGridDistortiond(
        keys=("image", "mask"), prob=0.5, distort_limit=(-0.01, 0.01), mode="nearest"),
])

transforms_valid = transforms.Compose([
])

# image loading utilities


def load_dicom(image_path):
    dicom = pydicom.read_file(image_path)
    img_arr = dicom.pixel_array
    img_arr = cv.resize(
        img_arr, (Config.image_sizes[0], Config.image_sizes[1]), interpolation=cv.INTER_AREA)
    return img_arr


def load_all_dicom_images(folder_path):
    t_paths = sorted(glob(os.path.join(folder_path, "*")),
                     key=lambda x: int(x.split('/')[-1].split(".")[0]))
    n_scans = len(t_paths)
    indices = np.quantile(list(range(n_scans)), np.linspace(
        0., 1., Config.image_sizes[2])).round().astype(int)
    t_paths = [t_paths[i] for i in indices]

    images = []
    for filename in t_paths:
        images.append(load_dicom(filename))
    images = np.stack(images, -1)
    images = images - np.min(images)
    images = images / (np.max(images) + 1e-4)
    images = (images * 255).astype(np.uint8)

    return images


def load_sample(row, has_mask=True):
    image = load_all_dicom_images(row.image_folder)
    if image.ndim < 4:
        image = np.expand_dims(image, 0).repeat(3, 0)  # to 3ch

    if has_mask:
        mask_org = nib.load(row.mask_file).get_fdata()
        shape = mask_org.shape
        mask_org = mask_org.transpose(1, 0, 2)[::-1, :, ::-1]  # (d, w, h)
        mask = np.zeros((7, shape[0], shape[1], shape[2]))
        for cid in range(7):
            mask[cid] = (mask_org == (cid+1))
        mask = mask.astype(np.uint8) * 255
        mask = Resize(Config.image_sizes)(mask).numpy()

        return image, mask
    else:
        return image

# converting 2D model to 3D


def convert_3d(module):
    module_output = module

    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = torch.nn.BatchNorm3d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    elif isinstance(module, Conv2dSame):
        module_output = Conv3dSame(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
        )
        module_output.weight = torch.nn.Parameter(
            module.weight.unsqueeze(-1).repeat(1, 1, 1, 1, module.kernel_size[0]))

    elif isinstance(module, torch.nn.Conv2d):
        module_output = torch.nn.Conv3d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode
        )
        module_output.weight = torch.nn.Parameter(
            module.weight.unsqueeze(-1).repeat(1, 1, 1, 1, module.kernel_size[0]))

    elif isinstance(module, torch.nn.MaxPool2d):
        module_output = torch.nn.MaxPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.AvgPool2d):
        module_output = torch.nn.AvgPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            ceil_mode=module.ceil_mode,
        )

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_3d(child)
        )

    del module
    return module_output

# mixup augementaion to avoid overfitting


def mixup(input, truth, clip=[0, 1]):
    """
    perform mixup data augmentation technique 
    """
    indices = torch.randperm(input.size(0))
    shuffled_input = input[indices]
    shuffled_labels = truth[indices]

    lam = np.random.uniform(clip[0], clip[1])
    input = input * lam + shuffled_input * (1 - lam)
    return input, truth, shuffled_labels, lam

# getting system device


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     return torch.device("mps")
    else:
        return torch.device("cpu")
