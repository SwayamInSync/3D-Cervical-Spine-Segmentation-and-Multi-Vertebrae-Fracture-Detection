from dataclasses import dataclass
import multiprocessing


@dataclass
class Config:
    DEBUG = False
    n_blocks = 4
    n_folds = 5
    backbone = "tf_efficientnetv2_s_in21ft1k"
    kernel_type = "efficient-unet-segmentation"

    image_sizes = [128, 128, 128]
    init_lr = 3e-3
    batch_size = 4
    drop_rate = 0.
    drop_path_rate = 0.
    loss_weights = [1, 1]
    p_mixup = 0.1

    data_dir = "stage-1/files/data"
    use_amp = True
    num_workers = multiprocessing.cpu_count()
    out_dim = 7

    n_epochs = 1000

    model_dir = 'stage-1/files/models'


if __name__ == "__main__":
    print(Config.num_workers)
