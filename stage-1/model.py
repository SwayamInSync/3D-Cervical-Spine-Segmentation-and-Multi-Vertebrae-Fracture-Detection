import timm
import torch.nn as nn
from config import Config
import segmentation_models_pytorch as smp


class SegModel(nn.Module):
    def __init__(self, backbone, segtype='unet', pretrained=False):
        super(SegModel, self).__init__()
        self.drop_rate = Config.drop_rate
        self.drop_path_rate = Config.drop_path_rate
        self.n_blocks = Config.n_blocks
        self.out_dim = Config.out_dim

        self.encoder = timm.create_model(
            backbone,
            in_chans=3,
            features_only=True,
            drop_rate=self.drop_rate,
            drop_path_rate=self.drop_path_rate,
            pretrained=pretrained
        )
#         g = self.encoder(torch.rand(1, 3, 64, 64))
#         encoder_channels = [1] + [_.shape[1] for _ in g]
        encoder_channels = [1] + [stage["num_chs"]
                                  for stage in self.encoder.feature_info]
        decoder_channels = [256, 128, 64, 32, 16]
        if segtype == 'unet':
            self.decoder = smp.unet.decoder.UnetDecoder(
                encoder_channels=encoder_channels[:self.n_blocks+1],
                decoder_channels=decoder_channels[:self.n_blocks],
                n_blocks=self.n_blocks,
            )

        self.segmentation_head = nn.Conv2d(
            decoder_channels[self.n_blocks-1], self.out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        global_features = [0] + self.encoder(x)[:self.n_blocks]
        seg_features = self.decoder(*global_features)
        seg_features = self.segmentation_head(seg_features)
        return seg_features


if __name__ == "__main__":
    import torch
    from utils import convert_3d

    m = SegModel(Config.backbone)
    m = convert_3d(m)
    op = m(torch.rand(1, 3, 128, 128, 128))
    assert op.shape == (1, 7, 128, 128, 128)
    print("Success")
