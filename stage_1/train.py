import torch
import random
from tqdm import tqdm
from torch.cuda import amp
import numpy as np
import torch.optim as optim
import wandb
import os
import gc

from .config import Config
from .utils import mixup, get_device, convert_3d, transforms_train, transforms_valid
from .loss_and_metric import multilabel_dice_score, dice_loss
from .model import SegModel
from .dataset import SEGDataset, get_dataframe


def train_func(model, optimizer, train_loader, criterion, device, scaler=None):
    model.train()
    train_loss = []
    bar = tqdm(train_loader)

    for images, gt_masks in bar:
        optimizer.zero_grad()
        images = images.to(device)
        gt_masks = gt_masks.to(device)

        do_mixup = False
        if random.random() < Config.p_mixup:
            do_mixup = True
            images, gt_masks, shuffled_gt_masks, lambda_ = mixup(
                images, gt_masks)

        # only available for cuda device
        if device == "cuda":
            with amp.autocast():
                logits = model(images)
                loss = criterion(logits, gt_masks)

                if do_mixup:
                    loss2 = criterion(logits, shuffled_gt_masks)
                    loss = loss*lambda_ + loss2*(1-lambda_)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, gt_masks)

            if do_mixup:
                loss2 = criterion(logits, shuffled_gt_masks)
                loss = loss*lambda_ + loss2*(1-lambda_)

            loss.backward()
            optimizer.step()
        train_loss.append(loss.item())
        bar.set_description(f'Train Loss:{np.mean(train_loss[-30:]):.4f}')

    return np.mean(train_loss)


def valid_func(model, loader_valid, criterion, device):
    model.eval()
    valid_loss = []
    outputs = []
    ths = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    batch_metrics = [[]] * 7
    bar = tqdm(loader_valid)
    with torch.no_grad():
        for images, gt_masks in bar:
            images = images.to(device)
            gt_masks = gt_masks.to(device)

            logits = model(images)
            loss = criterion(logits, gt_masks)
            valid_loss.append(loss.item())
            for thi, th in enumerate(ths):
                pred = (logits.sigmoid() > th).float().detach()
                for i in range(logits.shape[0]):
                    tmp = multilabel_dice_score(
                        y_pred=logits[i].sigmoid().cpu(),
                        y_true=gt_masks[i].cpu(),
                        threshold=0.5,
                    )
                    batch_metrics[thi].extend(tmp)
            bar.set_description(f'Valid Loss:{np.mean(valid_loss[-30:]):.4f}')

    metrics = [np.mean(this_metric) for this_metric in batch_metrics]
    print('best th:', ths[np.argmax(metrics)], 'best dc:', np.max(metrics))

    return np.mean(valid_loss), np.max(metrics)


def run(fold):
    device = get_device()
    os.makedirs(Config.model_dir, exist_ok=True)

    wandb.init(project='Cervical Spine Fracture Detection', name=f'Fold_{fold}', config={
               'batch_size': Config.batch_size, 'learning_rate': Config.init_lr})

    model_file = os.path.join(
        Config.model_dir, f'{Config.kernel_type}_fold{fold}_best.pth')

    df_seg = get_dataframe()
    train_ = df_seg[df_seg['fold'] != fold].reset_index(drop=True)
    valid_ = df_seg[df_seg['fold'] == fold].reset_index(drop=True)

    dataset_train = SEGDataset(train_, 'train', transform=transforms_train)
    dataset_valid = SEGDataset(valid_, 'valid', transform=transforms_valid)
    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)
    loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)

    model = SegModel(Config.backbone)
    model = convert_3d(model)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=Config.init_lr)
    scaler = torch.cuda.amp.GradScaler()
    from_epoch = 0
    metric_best = 0.
    loss_min = np.inf
    criterion = dice_loss

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, Config.n_epochs)

    for epoch in range(1, Config.n_epochs+1):
        scheduler_cosine.step(epoch-1)

        train_loss = train_func(model, optimizer, loader_train,
                                criterion, device, scaler)
        valid_loss, metric = valid_func(model, loader_valid, criterion, device)

        wandb.log({
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'metric': metric,
            'learning_rate': optimizer.param_groups[0]["lr"]
        })

        if metric > metric_best:
            print(
                f'metric_best ({metric_best:.6f} --> {metric:.6f}). Saving model ...')
            torch.save(model.state_dict(), model_file)
            # Log model weights to W&B
            wandb.save(model_file)
            metric_best = metric

        # save last
        if not Config.DEBUG:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'score_best': metric_best,
                },
                model_file.replace('_best', '_last')
            )
    wandb.finish()
    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    # training on each fold and saving the best model to corresponding fold
    for i in range(Config.n_folds):
        run(i)
