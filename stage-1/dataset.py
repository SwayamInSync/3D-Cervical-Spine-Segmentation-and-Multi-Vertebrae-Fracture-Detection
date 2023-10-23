import os
import pandas as pd
from config import Config
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset
import numpy as np

from utils import load_sample


# dataset class
class SEGDataset(Dataset):
    def __init__(self, df, mode, transform) -> None:
        self.df = df.reset_index()
        self.mode = mode
        self.transform = transform
        self.revert_list = [
            '1.2.826.0.1.3680043.1363',
            '1.2.826.0.1.3680043.20120',
            '1.2.826.0.1.3680043.2243',
            '1.2.826.0.1.3680043.24606',
            '1.2.826.0.1.3680043.32071'
        ]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image, mask = load_sample(row, has_mask=True)

        if row.StudyInstanceUID in self.revert_list:
            mask = mask[:, :, :, ::-1]

        res = self.transform({'image': image, 'mask': mask})
        image = res['image'] / 255.
        mask = res['mask']
        mask = (mask > 127).astype(np.float32)

        image, mask = torch.tensor(image).float(), torch.tensor(mask).float()

        return image, mask


def fix_paths(row):
    id_ = os.path.basename(row['mask_file'])
    row['mask_file'] = os.path.join(Config.data_dir, "segmentations", id_)
    row['image_folder'] = os.path.join(
        Config.data_dir, "subset_dataset", id_[:-4])
    return row


def get_dataframe():
    df_train = pd.read_csv(os.path.join(Config.data_dir, 'data.csv'))
    df_train.drop("fold", inplace=True, axis=1)
    df_train.drop("Unnamed: 0", inplace=True, axis=1)

    df_seg = df_train.apply(fix_paths, axis=1)

    # applying k-fold
    kf = KFold(Config.n_folds)
    df_seg['fold'] = -1
    for fold, (train_idx, valid_idx) in enumerate(kf.split(df_seg, df_seg)):
        df_seg.loc[valid_idx, 'fold'] = fold

    return df_seg


if __name__ == "__main__":
    from utils import transforms_train
    import matplotlib.pyplot as plt

    Config.n_folds = 4
    df_seg = get_dataframe()
    dataset_show = SEGDataset(df_seg, 'train', transform=transforms_train)
    for i in range(2):
        f, axarr = plt.subplots(1, 2)
        for p in range(2):
            idx = i*2+p
            img, mask = dataset_show[idx]
            print(mask.shape)
            img = img[:, :, :, 60]  # picking the 60th image
            mask = mask[:, :, :, 60]  # picking corresponding mask
            print(mask.shape)
            print("*"*100)
            mask[0] = mask[0] + mask[3] + mask[6]
            mask[1] = mask[1] + mask[4]
            mask[2] = mask[2] + mask[5]
            mask = mask[:3]
            img = img * 0.7 + mask * 0.3
            axarr[p].imshow(img.transpose(0, 1).transpose(1, 2).squeeze())
    plt.show()
