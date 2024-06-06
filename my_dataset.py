import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import transforms as T
import torch
import pickle


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0, 0), std=(1, 1)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, psd, target):
        return self.transforms(psd, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0, 0), std=(1, 1)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, psd, target):
        return self.transforms(psd, target)


def get_transform(train, mean=(0, 0), std=(1, 1)):
    if train:
        return SegmentationPresetEval(mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def read_pkl(data_url):
    file = open(data_url, 'rb')
    content = pickle.load(file)
    file.close()
    return content


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()

        self.flag = "training" if train else "test"
        data_root = os.path.join(root, "RF_image", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        if self.flag == "training":
            self.psd_names = sorted([os.path.join(data_root, "RF_data_medfilt", '%s' % f) for f in os.listdir(os.path.join(data_root, "RF_data_medfilt")) if f.endswith(".pkl")])
            self.mask_names = sorted([os.path.join(data_root, "image_label", '%s' % f) for f in os.listdir(os.path.join(data_root, "image_label")) if f.endswith(".jpg")])
        else:
            self.psd_names = sorted([os.path.join(data_root, "RF_data_medfilt", '%s' % f) for f in os.listdir(os.path.join(data_root, "RF_data_medfilt")) if f.endswith(".pkl")])
            self.mask_names = sorted([os.path.join(data_root, "image_label", '%s' % f) for f in os.listdir(os.path.join(data_root, "image_label")) if f.endswith(".jpg")])

    def __getitem__(self, idx):
        psd_data = read_pkl(self.psd_names[idx])
        mask = Image.open(self.mask_names[idx]).convert('L')
        mask = Image.fromarray(np.array(mask))

        if self.transforms is not None:
            psd_data_, mask = self.transforms(psd_data, mask)
        return psd_data_, mask

    def __len__(self):
        return len(self.psd_names)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")
    parser.add_argument("--data_path", default="./data/", help="Image and PSD root")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")
    args = parse_args()
    mean = (-48.4187, -44.5439)
    std = (3.2451, 4.9254)

    train_dataset = DriveDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=4,
                                               num_workers=1,
                                               shuffle=True,
                                               pin_memory=True)