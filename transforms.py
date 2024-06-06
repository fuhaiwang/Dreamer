import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import os
import matplotlib.pyplot as plt


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, psd, target):
        for t in self.transforms:
            psd, target = t(psd, target)
        return psd, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class Resize(object):
    def __init__(self, x_size, y_size):
        self.x_size = x_size
        self.y_size = y_size

    def __call__(self, psd, target):
        target = F.resize(target, (self.x_size, self.y_size), interpolation=T.InterpolationMode.NEAREST)

        return psd, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)

        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)

        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, psd, target):
        psd = torch.as_tensor(psd, dtype=torch.float)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return psd, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, psd, target):
        psd = F.normalize(psd, mean=self.mean, std=self.std)
        target = target.float().div(255)
        return psd, target


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    image = torch.randn(4, 1, 38, 15)
    image = image.to(device)

    mask = Image.open(os.path.join("./data/RF_Image/training/image_label_integrated_1205/"
                                                          "pred_result_6.png")).convert('L')

    T_resize = Resize(38, 15)
    mask_, mask = T_resize(mask, mask)

    plt.imshow(mask)
    plt.show()
    plt.imshow(mask_)
    plt.show()























