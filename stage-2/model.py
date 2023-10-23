import timm
import torch
import torch.nn as nn

from config import Config


class CLSModel(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super(TimmModel, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            in_chans=Config.in_chans,
            num_classes=Config.out_dim,
            features_only=False,
            drop_rate=Config.drop_rate,
            drop_path_rate=Config.drop_path_rate,
            pretrained=pretrained
        )

        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'convnext' in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()

        self.lstm = nn.LSTM(hdim, 256, num_layers=2,
                            dropout=Config.drop_rate, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(Config.drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(256, Config.out_dim),
        )

    def forward(self, x):  # (bs, nslice, ch, sz, sz)
        bs = x.shape[0]
        x = x.view(bs * Config.n_slice_per_c, Config.in_chans, Config.image_size,
                   Config.image_size)  # (bs*nslice, ch, sz, sz)
        feat = self.encoder(x)
        feat = feat.view(bs, Config.n_slice_per_c, -1)
        feat, _ = self.lstm(feat)
        feat = feat.contiguous().view(bs * Config.n_slice_per_c, -1)
        feat = self.head(feat)
        feat = feat.view(bs, Config.n_slice_per_c).contiguous()

        return feat


if __name__ == "__main__":
    m = CLSModel(Config.backbone)
    op = m(torch.rand(2, Config.n_slice_per_c, Config.in_chans,
                      Config.image_size, Config.image_size))
    assert op.shape == (2, 15)
    print("Success")
