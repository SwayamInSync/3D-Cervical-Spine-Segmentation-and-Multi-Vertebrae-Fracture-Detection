import os
import pandas as pd
import numpy as np
from glob import glob
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


from stage_1.utils import get_device, load_dicom, convert_3d
from stage_1.model import SegModel
from stage_2.model import CLSModel
import stage_1.config as SegConfig
import stage_2.config as CLSConfig
from cls_data_processing import load_cropped_images
from config import Config


def load_dicom_line_par(path):
    t_paths = sorted(glob(os.path.join(path, "*")),
                     key=lambda x: int(x.split('/')[-1].split(".")[0]))

    n_scans = len(t_paths)

    indices = np.quantile(list(range(n_scans)), np.linspace(
        0., 1., Config.image_size_seg[2])).round().astype(int)
    t_paths = [t_paths[i] for i in indices]

    images = []
    for filename in t_paths:
        images.append(load_dicom(filename))
    images = np.stack(images, -1)

    images = images - np.min(images)
    images = images / (np.max(images) + 1e-4)
    images = (images * 255).astype(np.uint8)

    return images


def get_dataframe():
    df = pd.read_csv(os.path.join(Config.data_dir, 'test.csv'))
    if df.iloc[0].row_id == '1.2.826.0.1.3680043.10197_C1':
        # test_images and test.csv are inconsistent in the dev dataset, fixing labels for the dev run.
        df = pd.DataFrame({
            "row_id": ['1.2.826.0.1.3680043.22327_C1', '1.2.826.0.1.3680043.25399_C1', '1.2.826.0.1.3680043.5876_C1'],
            "StudyInstanceUID": ['1.2.826.0.1.3680043.22327', '1.2.826.0.1.3680043.25399', '1.2.826.0.1.3680043.5876'],
            "prediction_type": ["C1", "C1", "patient_overall"]}
        )
    df = pd.DataFrame({
        'StudyInstanceUID': df['StudyInstanceUID'].unique().tolist()
    })
    df['image_folder'] = df['StudyInstanceUID'].apply(
        lambda x: os.path.join(Config.data_dir, x))

    return df


class SegTestDataset(Dataset):

    def __init__(self, df):
        self.df = df.reset_index()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image = load_dicom_line_par(row.image_folder)
        if image.ndim < 4:
            image = np.expand_dims(image, 0)
        image = image.astype(np.float32).repeat(3, 0)  # to 3ch
        image = image / 255.
        return torch.tensor(image).float()


# load models

def load_seg_models(model_dir_seg, device):
    models_seg = []

    backbone = SegConfig.Config.backbone
    model_dir_seg = 'seg_models'
    n_blocks = 4
    for fold in range(SegConfig.Config.n_folds):
        model = SegModel(backbone, pretrained=False)
        model = convert_3d(model)
        model = model.to(device)
        load_model_file = os.path.join(
            model_dir_seg, f'fold_{fold}.pth')
        sd = torch.load(load_model_file, map_location=device)
        if 'model_state_dict' in sd.keys():
            sd = sd['model_state_dict']
        sd = {k[7:] if k.startswith('module.') else k: sd[k]
              for k in sd.keys()}
        model.load_state_dict(sd, strict=True)
        model.eval()
        models_seg.append(model)
    return models_seg

# load classification models


def load_cls_models(model_dir_cls, device):
    backbone = CLSConfig.Config.backbone
    in_chans = 6
    models_cls = []

    for fold in range(CLSConfig.Config.n_folds):
        model = CLSModel(backbone, pretrained=False)
        load_model_file = os.path.join(
            model_dir_cls, f'fold_{fold}.pth')
        sd = torch.load(load_model_file, map_location=device)
        if 'model_state_dict' in sd.keys():
            sd = sd['model_state_dict']
        sd = {k[7:] if k.startswith('module.') else k: sd[k]
              for k in sd.keys()}
        model.load_state_dict(sd, strict=True)
        model = model.to(device)
        model.eval()
        models_cls.append(model)
    return models_cls


# predict


def predict(df, loader_seg, models_cls, models_seg):
    outputs1 = []

    bar = tqdm(loader_seg)
    with torch.no_grad():
        for batch_id, (images) in enumerate(bar):
            images = images.to(device)

            # SEG
            pred_masks = []
            for model in models_seg:
                pmask = model(images).sigmoid()
                pred_masks.append(pmask)
            pred_masks = torch.stack(pred_masks, 0).mean(0).cpu().numpy()
            # np.save("pred_masks", pred_masks)

            # Build cls input
            cls_inp = []
            threads = [None] * 7
            cropped_images = [None] * 7

            for i in range(pred_masks.shape[0]):
                row = df.iloc[batch_id*Config.batch_size_seg+i]
                cropped_images = load_cropped_images(
                    pred_masks[i], row.image_folder, threads, cropped_images)
                cls_inp.append(cropped_images.permute(
                    0, 3, 1, 2).float() / 255.)
            cls_inp = torch.stack(cls_inp, 0).to(
                device)  # (1, 105, 6, 224, 224)

            pred_cls1 = []

            # CLS 1
            cls_inp = cls_inp.view(7, 15, 6, Config.image_size_cls,
                                   Config.image_size_cls).contiguous()
            for _, model in enumerate(models_cls):
                logits = model(cls_inp)
                pred_cls1.append(
                    logits.sigmoid().view(-1, 7, Config.n_slice_per_c))

            pred_cls1 = torch.stack(pred_cls1, 0).mean(0)
            outputs1.append(pred_cls1.cpu())
    outputs1 = torch.cat(outputs1)
    preds = outputs1.mean(-1).clamp(0.0001, 0.9999)
    return preds


if __name__ == "__main__":
    device = get_device()

    df = get_dataframe().head(1)

    dataset_seg = SegTestDataset(df)
    loader_seg = torch.utils.data.DataLoader(
        dataset_seg, batch_size=Config.batch_size_seg, shuffle=False, num_workers=Config.num_workers)

    models_seg = load_seg_models("seg_models", device)
    models_cls = load_cls_models("cls_models", device)
    preds = predict(df, loader_seg, models_cls, models_seg)
    row_ids = []
    for _, row in df.iterrows():
        for i in range(7):
            row_ids.append(row.StudyInstanceUID + f'_C{i+1}')
    df_sub = pd.DataFrame({
        'row_id': row_ids,
        'fractured': preds.view(-1)})
    df_sub.to_csv("result.csv", index=False)
