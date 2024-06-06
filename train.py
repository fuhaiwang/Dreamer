import os
import time
import datetime

import torch

from src.PSD2Image import psd2image
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import DriveDataset
import transforms as T


class SegmentationPresetEval:
    def __init__(self,  mean=(0, 0), std=(1, 1)):
        self.transforms = T.Compose([
            T.Resize(1080, 980),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, psd, target):
        return self.transforms(psd, target)


def get_transform(train,  mean=(0, 0), std=(1, 1)):
    if train:
        return SegmentationPresetEval(mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def create_model():
    model = psd2image()
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu") 
    torch.cuda.set_device(0)
    batch_size = args.batch_size
    num_classes = args.num_classes + 1 

    # using compute_mean_std.py
    mean = (-49.3247, -45.7729)
    std = (3.8249, 5.7447) 

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    train_dataset = DriveDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = DriveDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 8])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True)

    model = create_model()
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=False)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer']) 
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler']) 
        args.start_epoch = checkpoint['epoch'] + 1 
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"]) 

    best_dice = 0.
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         # f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + "\n\n")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best is True:
            torch.save(save_file, "pth/best_model.pth")
        else:
            torch.save(save_file, "pth/model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data_path", default="./Dataset/", help="Image and PSD root")
    # exclude background
    parser.add_argument("--num_classes", default=1, type=int)
    parser.add_argument("--device", default="cuda:0", help="training device")
    parser.add_argument("-b", "--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=100, type=int, metavar="N",
                        help="number of total epochs to train")
    
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print_freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save_best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)