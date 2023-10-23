from dataclasses import dataclass
import multiprocessing


@dataclass
class Config:
    kernel_type = 'efficientnet-lstm-classification'

    n_folds = 5
    backbone = 'tf_efficientnetv2_s_in21ft1k'

    image_size = 224
    n_slice_per_c = 15
    in_chans = 6

    init_lr = 23e-5
    eta_min = 23e-6  # schedulers
    batch_size = 8
    drop_rate = 0.
    drop_rate_last = 0.3
    drop_path_rate = 0.
    p_mixup = 0.5
    p_rand_order_v1 = 0.2

    data_dir = 'stage-2/data'
    use_amp = True
    num_workers = multiprocessing.cpu_count()
    out_dim = 1

    n_epochs = 75

    model_dir = './models'
