import os
import time

import torch
from PIL import Image
from src.PSD2Image import psd2image

from train import get_transform
import train_utils.train_and_eval as eval
import numpy
import numpy as np
import pickle
from tqdm import tqdm
import scipy.io


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def read_pkl(data_url):
    file = open(data_url, 'rb')
    content = pickle.load(file)
    file.close()
    return content


def main():
    total_loss = 0
    total_loss_mae = 0
    total_loss_F_beta = 0
    total_loss_BCE = 0
    total_loss_IoU = 0

    temp = 0

    weights_path = "./pretrained/best_model.pth"

    # training  test
    img_label_path = "./Dataset/RF_image/test/image_label"
    psd_path = "./Dataset/RF_image/test/RF_data_medfilt"
    save_path = "./result/test/"

    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_label_path), f"image {img_label_path} not found."
    assert os.path.exists(psd_path), f"psd {psd_path} not found."

    mean = (-49.3247, -45.7729)
    std = (3.8249, 5.7447)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create model
    model = psd2image()

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # label
    file_names_image = numpy.array(sorted(os.listdir(img_label_path)))
    file_names_psd = numpy.array(sorted(os.listdir(psd_path)))
    loss_SSIM_all = torch.empty((file_names_image.shape[0], 1), dtype=torch.tensor([]).dtype)
    loss_mae_all = torch.empty((file_names_image.shape[0], 1), dtype=torch.tensor([]).dtype)
    loss_F_beta_all = torch.empty((file_names_image.shape[0], 1), dtype=torch.tensor([]).dtype)
    loss_BCE_all = torch.empty((file_names_image.shape[0], 1), dtype=torch.tensor([]).dtype)
    loss_IoU_all = torch.empty((file_names_image.shape[0], 1), dtype=torch.tensor([]).dtype)

    for image_file_name in tqdm(file_names_image):

        img_label_name_path = os.path.join(img_label_path, image_file_name)
        psd_name_path = os.path.join(psd_path, file_names_psd[temp])

        original_img = Image.open(img_label_name_path).convert('L')
        psd_data = read_pkl(psd_name_path)

        transforms = get_transform(train=True, mean=mean, std=std)
        if transforms is not None:
            psd_data, mask = transforms(psd_data, original_img)

        model.eval()
        with torch.no_grad():

            output = model(torch.unsqueeze(psd_data, dim=0).to(device))
            output = torch.abs(output)
            output = torch.squeeze(output, dim=0)
            mask = mask.to(device)

            loss_ssim = eval.criterion_SSIM(output, torch.unsqueeze(mask, dim=0))

            loss_mae = eval.compute_mae(output, torch.unsqueeze(mask, dim=0),)
            loss_F_beta = eval.compute_F_beta(output, torch.unsqueeze(mask, dim=0))
            loss_BCE = eval.compute_bce(output, torch.unsqueeze(mask, dim=0))
            loss_IoU = eval.compute_IoU(output, torch.unsqueeze(mask, dim=0))

            total_loss += loss_ssim

            total_loss_mae += loss_mae
            total_loss_F_beta += loss_F_beta
            total_loss_BCE += loss_BCE
            total_loss_IoU += loss_IoU

            loss_SSIM_all[temp] = loss_ssim
            loss_mae_all[temp] = loss_mae
            loss_F_beta_all[temp] = loss_F_beta
            loss_BCE_all[temp] = loss_BCE
            loss_IoU_all[temp] = loss_IoU
            prediction = torch.squeeze(output, dim=0)

            prediction_ = prediction.to("cpu").numpy().astype(np.uint8) * 255

            prediction_gray_image = Image.fromarray(prediction_, 'L')
            prediction_gray_image.save(os.path.join(save_path, '{}_prediction.png'.format(image_file_name)))

            mask = mask.to("cpu").numpy().astype(np.uint8)*255
            mask_gray_image = Image.fromarray(mask, 'L')
            mask_gray_image.save(os.path.join(save_path, image_file_name))

        temp += 1

    print("SSIM", 1-total_loss/temp)
    print("MAE", total_loss_mae/temp)
    print("F_beta", 1-total_loss_F_beta/temp)
    print("BCE", total_loss_BCE/temp)
    print("IoU", 1-total_loss_IoU/temp)

    print("SSIM std", torch.std(loss_SSIM_all))
    print("mae std", torch.std(loss_mae_all))
    print("F beta std", torch.std(loss_F_beta_all))
    print("BCE std", torch.std(loss_BCE_all))
    print("IoU std", torch.std(1-loss_IoU_all))

    scipy.io.savemat('./result/loss_SSIM_all.mat', {'loss_SSIM_all': loss_SSIM_all.numpy()})


if __name__ == '__main__':
    main()



