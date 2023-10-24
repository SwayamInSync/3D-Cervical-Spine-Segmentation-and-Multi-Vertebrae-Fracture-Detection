import pandas as pd
import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .config import Config


class CLSDataset(Dataset):
    def __init__(self, df, mode, transform):

        self.df = df.reset_index()
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        cid = row.c  # cid basically is the which bone

        images = []

        for ind in list(range(Config.n_slice_per_c)):
            filepath = os.path.join(
                Config.data_dir, f'{row.StudyInstanceUID}_{cid}_{ind}.npy')
            image = np.load(filepath)
            image = self.transform(image=image)['image']
            image = image.transpose(2, 0, 1).astype(np.float32) / 255.
            images.append(image)
        images = np.stack(images, 0)

        if self.mode != 'test':
            images = torch.tensor(images).float()
            labels = torch.tensor(
                [row.label] * Config.n_slice_per_c).float()  # extending label

            if self.mode == 'train' and random.random() < Config.p_rand_order_v1:
                indices = torch.randperm(images.size(0))
                images = images[indices]

            return images, labels
        else:
            return torch.tensor(images).float()


def get_dataframe():
    df_seg = pd.read_csv(os.path.join(Config.data_dir, "train_seg.csv"))
    sid = []
    cs = []
    label = []
    fold = []
    for _, row in df_seg.iterrows():
        for i in [1, 2, 3, 4, 5, 6, 7]:
            sid.append(row.StudyInstanceUID)
            cs.append(i)
            label.append(row[f'C{i}'])
            fold.append(row.fold)

    df = pd.DataFrame({
        'StudyInstanceUID': sid,
        'c': cs,
        'label': label,
        'fold': fold
    })
