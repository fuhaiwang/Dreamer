import torch
import torch.nn.functional as F
import train_utils.distributed_utils as utils


def mask_normalize(mask):
# input 'mask': HxW
# output: HxW [0,255]
    return mask/(torch.max(mask)+1e-8)


def compute_mae(pred_, gt):
# input 'mask1': HxW or HxWxn (asumme that all the n channels are the same and only the first channel will be used)
#       'mask2': HxW or HxWxn
# output: a value MAE, Mean Absolute Error
    num_images = len(pred_)
    total_maeError = 0
    for i in range(num_images):
        mask1 = pred_[i]
        mask2 = gt[i]
        h, w = mask1.shape[0], mask1.shape[1]
        mask1 = mask_normalize(mask1)
        mask2 = mask_normalize(mask2)
        sumError = torch.sum(torch.abs((mask1.float() - mask2.float())))
        maeError = sumError/(float(h)*float(w)+1e-12)
        total_maeError += maeError

    avg_maeError = total_maeError/num_images
    return avg_maeError   # MAE


def compute_mse(pred_, gt): 
    loss_MSE = F.mse_loss(pred_, gt)
    return loss_MSE 


def compute_IoU(pred, mask):

    inter = (pred*mask).sum(dim=(1, 2))
    union = (pred+mask).sum(dim=(1, 2))
    iou = 1-(inter+0.0001)/(union-inter+0.0001)
    return iou.mean()


def compute_bce(pred, mask):
    bce = F.binary_cross_entropy(pred, mask, reduce='none')

    return bce.mean()



def compute_F_beta(y_pred, y_true, threshold = 0.5, beta=1., epsilon=1e-6):
    """
    Differentiable F-beta score

    F_beta = (1 + beta^2) * TP / [beta^2 * (TP + FN) + (TP + FP)]
    = (1 + beta^2) (true * pred) / [beta^2 * true + pred]

    Using this formula, we don't have to use precision and recall.

    F1 score is an example of F-beta score with beta=1.
    """
    y_pred = (y_pred > threshold).float()
    y_true = (y_true > threshold).float()

    TP = (y_true * y_pred).sum().float()
    FP = ((y_true - y_pred) == 1).sum().float()
    FN = ((y_true - y_pred) == -1).sum().float()

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    beta2 = beta ** 2   # beta squared
    f_beta_score = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)

    return 1 - f_beta_score


def criterion_SSIM(pred_, gt):
    num_images = len(pred_)
    total_ssim = 0
    for i in range(num_images):
        pred = pred_[i]
        target = gt[i]

        gray_pred = pred
        gray_target = target.float()

        mean_pred, var_pred = torch.mean(gray_pred), torch.var(gray_pred)
        mean_target, var_target = torch.mean(gray_target), torch.var(gray_target)
        covar = torch.mean((gray_pred - mean_pred) * (gray_target - mean_target))

        k1 = 0.01
        k2 = 0.03
        L = 1

        c1 = (k1 * L) ** 2
        c2 = (k2 * L) ** 2
        c3 = c2 / 2
        numerator = (2 * mean_pred * mean_target + c1) * (2 * covar + c2)
        denominator = (mean_pred ** 2 + mean_target ** 2 + c1) * (var_pred + var_target + c2)
        ssim = numerator / denominator

        total_ssim += ssim

    avg_ssim = total_ssim / num_images
    if 1 - avg_ssim == None:
        return 1 - avg_ssim

    return 1 - avg_ssim


def criterion_SSIM_MSE(pred_, gt):
    num_images = len(pred_)
    total_ssim = 0
    MSE_loss_alpha = 0.5   # 0.8
    SSIM_loss_beta = 1-MSE_loss_alpha   # 0.2

    for i in range(num_images):
        pred = pred_[i]
        target = gt[i]
        gray_pred = pred
        gray_target = target.float()
        mean_pred, var_pred = torch.mean(gray_pred), torch.var(gray_pred)
        mean_target, var_target = torch.mean(gray_target), torch.var(gray_target)
        covar = torch.mean((gray_pred - mean_pred) * (gray_target - mean_target))
        k1 = 0.01
        k2 = 0.03
        L = 1
        c1 = (k1 * L) ** 2
        c2 = (k2 * L) ** 2
        c3 = c2 / 2
        numerator = (2 * mean_pred * mean_target + c1) * (2 * covar + c2)
        denominator = (mean_pred ** 2 + mean_target ** 2 + c1) * (var_pred + var_target + c2)
        ssim = numerator / denominator
        total_ssim += ssim

    avg_ssim = total_ssim / num_images

    # MSE
    loss_MSE = F.mse_loss(pred_, gt)
    if loss_MSE + 1 - avg_ssim == None:
        return loss_MSE + 1 - avg_ssim

    return loss_MSE * MSE_loss_alpha + (1 - avg_ssim) * SSIM_loss_beta


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            # image, target = image.to(device), target.to(device)
            image, target = image.cuda(), target.cuda()
            output = model(image)
            dice.update(output, target)

        dice.reduce_from_all_processes()

    return dice.value.item()


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")  #
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    loss_weights = [0.35, 0.2, 0.45]

    for psd, target in metric_logger.log_every(data_loader, print_freq, header):
        psd, target = psd.cuda(), target.cuda()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(psd)
            loss_SSIM = criterion_SSIM(torch.squeeze(output), target)  # SSIM
            loss_mae = compute_mae(torch.squeeze(output), target)  # MSE MAE
            loss = loss_SSIM + loss_mae

        print(f'loss:{loss}')
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(f'lr:{lr}')
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
