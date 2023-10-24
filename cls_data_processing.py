import torch
import numpy as np
import cv2
import pydicom
from glob import glob
import threading
import os

from config import Config


def load_bone(msk, cid, t_paths, cropped_images):
    n_scans = len(t_paths)
    bone = []
    try:
        msk_b = msk[cid] > 0.2
        msk_c = msk[cid] > 0.05

        x = np.where(msk_b.sum(1).sum(1) > 0)[0]
        y = np.where(msk_b.sum(0).sum(1) > 0)[0]
        z = np.where(msk_b.sum(0).sum(0) > 0)[0]

        if len(x) == 0 or len(y) == 0 or len(z) == 0:
            x = np.where(msk_c.sum(1).sum(1) > 0)[0]
            y = np.where(msk_c.sum(0).sum(1) > 0)[0]
            z = np.where(msk_c.sum(0).sum(0) > 0)[0]

        x1, x2 = max(0, x[0] - 1), min(msk.shape[1], x[-1] + 1)
        y1, y2 = max(0, y[0] - 1), min(msk.shape[2], y[-1] + 1)
        z1, z2 = max(0, z[0] - 1), min(msk.shape[3], z[-1] + 1)
        zz1, zz2 = int(z1 / Config.msk_size *
                       n_scans), int(z2 / Config.msk_size * n_scans)

        inds = np.linspace(zz1, zz2-1, Config.n_slice_per_c).astype(int)
        inds_ = np.linspace(z1, z2-1, Config.n_slice_per_c).astype(int)
        for sid, (ind, ind_) in enumerate(zip(inds, inds_)):

            msk_this = msk[cid, :, :, ind_]

            images = []
            for i in range(-Config.n_ch//2+1, Config.n_ch//2+1):
                try:
                    dicom = pydicom.read_file(t_paths[ind+i])
                    images.append(dicom.pixel_array)
                except:
                    images.append(np.zeros((512, 512)))

            data = np.stack(images, -1)
            data = data - np.min(data)
            data = data / (np.max(data) + 1e-4)
            data = (data * 255).astype(np.uint8)
            msk_this = msk_this[x1:x2, y1:y2]
            xx1 = int(x1 / Config.msk_size * data.shape[0])
            xx2 = int(x2 / Config.msk_size * data.shape[0])
            yy1 = int(y1 / Config.msk_size * data.shape[1])
            yy2 = int(y2 / Config.msk_size * data.shape[1])
            data = data[xx1:xx2, yy1:yy2]
            data = np.stack([cv2.resize(data[:, :, i], (Config.image_size_cls, Config.image_size_cls),
                            interpolation=cv2.INTER_LINEAR) for i in range(Config.n_ch)], -1)
            msk_this = (msk_this * 255).astype(np.uint8)
            msk_this = cv2.resize(
                msk_this, (Config.image_size_cls, Config.image_size_cls), interpolation=cv2.INTER_LINEAR)

            data = np.concatenate([data, msk_this[:, :, np.newaxis]], -1)

            bone.append(torch.tensor(data))

    except:
        for sid in range(Config.n_slice_per_c):
            bone.append(torch.ones(
                (Config.image_size_cls, Config.image_size_cls, Config.n_ch+1)).int())

    cropped_images[cid] = torch.stack(bone, 0)


def load_cropped_images(msk, image_folder, threads, cropped_images):

    t_paths = sorted(glob(os.path.join(image_folder, "*")),
                     key=lambda x: int(x.split('/')[-1].split(".")[0]))
    for cid in range(7):
        threads[cid] = threading.Thread(
            target=load_bone, args=(msk, cid, t_paths, cropped_images))
        threads[cid].start()
    for cid in range(7):
        threads[cid].join()

    return torch.cat(cropped_images, 0)
