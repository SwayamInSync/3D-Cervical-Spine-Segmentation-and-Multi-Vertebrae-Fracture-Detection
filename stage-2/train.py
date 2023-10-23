from tqdm import tqdm
import random
import torch
import numpy as np
import torch.cuda.amp as amp
import time
import wandb
import os
import gc
import torch.optim as optim

from config import Config
from loss_and_metric import criterion
from utils import mixup, get_device, transforms_train, transforms_valid
from datasets import CLSDataset, get_dataframe
from model import CLSModel


def train_func(model, loader_train, optimizer, scaler=None):
    model.train()
    train_loss = []
    bar = tqdm(loader_train)
    for images, targets in bar:
        optimizer.zero_grad()
        images = images.cuda()
        targets = targets.cuda()

        do_mixup = False
        if random.random() < Config.p_mixup:
            do_mixup = True
            images, targets, targets_mix, lam = mixup(images, targets)

        with amp.autocast():
            logits = model(images)
            loss = criterion(logits, targets)
            if do_mixup:
                loss11 = criterion(logits, targets_mix)
                loss = loss * lam + loss11 * (1 - lam)
        train_loss.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bar.set_description(f'Train Loss:{np.mean(train_loss[-30:]):.4f}')

    return np.mean(train_loss)


def valid_func(model, loader_valid):
    model.eval()
    valid_loss = []
    gts = []
    outputs = []
    bar = tqdm(loader_valid)
    with torch.no_grad():
        for images, targets in bar:
            images = images.cuda()
            targets = targets.cuda()

            logits = model(images)
            loss = criterion(logits, targets)

            gts.append(targets.cpu())
            outputs.append(logits.cpu())
            valid_loss.append(loss.item())

            bar.set_description(f'Valid Loss:{np.mean(valid_loss[-30:]):.4f}')

    outputs = torch.cat(outputs)
    gts = torch.cat(gts)
    valid_loss = criterion(outputs, gts).item()

    return valid_loss


def run(fold):
    device = get_device()
    os.makedirs(Config.model_dir, exist_ok=True)

    wandb.init(project='Cervical Spine Fracture Detection', name=f'CLS_Fold_{fold}', config={
               'batch_size': Config.batch_size, 'learning_rate': Config.init_lr})

    model_file = os.path.join(
        Config.model_dir, f'{Config.kernel_type}_fold{fold}_best.pth')

    df = get_dataframe()
    train_ = df[df['fold'] != fold].reset_index(drop=True)
    valid_ = df[df['fold'] == fold].reset_index(drop=True)
    dataset_train = CLSDataset(train_, 'train', transform=transforms_train)
    dataset_valid = CLSDataset(valid_, 'valid', transform=transforms_valid)
    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers, drop_last=True)
    loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)

    model = CLSModel(Config.backbone, pretrained=True)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=Config.init_lr)
    scaler = torch.cuda.amp.GradScaler() if Config.use_amp else None

    metric_best = np.inf
    loss_min = np.inf

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, Config.n_epochs, eta_min=Config.eta_min)

    print(len(dataset_train), len(dataset_valid))

    for epoch in range(1, Config.n_epochs+1):
        scheduler_cosine.step(epoch-1)

        print(time.ctime(), 'Epoch:', epoch)

        train_loss = train_func(model, loader_train, optimizer, scaler)
        valid_loss = valid_func(model, loader_valid)
        metric = valid_loss

        wandb.log({
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'metric': metric,
            'learning_rate': optimizer.param_groups[0]["lr"]
        })

        if metric < metric_best:
            print(
                f'metric_best ({metric_best:.6f} --> {metric:.6f}). Saving model ...')
            torch.save(model.state_dict(), model_file)
            # Log model weights to W&B
            wandb.save(model_file)
            metric_best = metric

        # Save Last
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
