from dataclasses import dataclass


@dataclass
class Config:
    data_dir = 'test-data'
    image_size_seg = (128, 128, 128)
    msk_size = image_size_seg[0]
    image_size_cls = 224
    n_slice_per_c = 15
    n_ch = 5

    batch_size_seg = 1
    num_workers = 2
