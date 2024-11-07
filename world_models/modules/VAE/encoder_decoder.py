import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


class Encoder(nn.Module):
    def __init__(self, in_channels, in_feature_width ,final_feature_width, stem_channels=256, num_repeat=2):
        super().__init__()
        self.num_repeat = num_repeat
        self.in_feature_width = int(in_feature_width)
        # -- stem
        backbone = []
        ## -- repeat layer
        if not num_repeat == 0:
            for _ in range(num_repeat):
                backbone.append(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False
                        )
                    )
        backbone.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=stem_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        )
        feature_width = self.in_feature_width//2
        channels = stem_channels
        backbone.append(nn.BatchNorm2d(stem_channels))
        backbone.append(nn.ReLU(inplace=True))

        ## -- Deep layers
        while True:
            if feature_width <= final_feature_width:
                break
            if not num_repeat == 0:
                for _ in range(num_repeat):
                    backbone.append(
                        nn.Conv2d(
                            in_channels=channels,
                            out_channels=channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False
                        )
                    )
            backbone.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels*2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            channels *= 2
            feature_width //= 2
            backbone.append(nn.BatchNorm2d(channels))
            backbone.append(nn.ReLU(inplace=True))


        self.backbone = nn.Sequential(*backbone)

        self.last_channels = channels
        self.final_feature_width = feature_width

    def forward(self, x):
        x = self.backbone(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_dim, last_channels, final_feature_width, recover_channels, recover_width, stem_channels=256, num_repeat=2):
        super().__init__()
        backbone = []
        self.recover_channels = int(recover_channels)
        self.recover_width = int(recover_width)
        # stem
        backbone.append(nn.Linear(in_dim, int(last_channels*final_feature_width*final_feature_width), bias=False))
        backbone.append(Rearrange('B (C H W) -> B C H W', C=last_channels, H=final_feature_width, W=final_feature_width))
        backbone.append(nn.BatchNorm2d(last_channels))
        backbone.append(nn.ReLU(inplace=True))
        # residual_layer
        # backbone.append(ResidualStack(last_channels, 1, last_channels//4))
        # layers
        channels = last_channels
        feat_width = final_feature_width
        while True:
            if channels == stem_channels:
                break
            backbone.append(
                nn.ConvTranspose2d(
                    in_channels=channels,
                    out_channels=channels//2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            channels //= 2
            feat_width *= 2

            if not num_repeat == 0:
                for _ in range(num_repeat):
                    backbone.append(
                        nn.ConvTranspose2d(
                            in_channels=channels,
                            out_channels=channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False
                        )
                    )
            backbone.append(nn.BatchNorm2d(channels))
            backbone.append(nn.ReLU(inplace=True))
            
        # recover layer
        backbone.append(
            nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=self.recover_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )
        feat_width *= 2
        if not num_repeat == 0:
            for _ in range(num_repeat):
                backbone.append(
                    nn.ConvTranspose2d(
                        in_channels=self.recover_channels,
                        out_channels=self.recover_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False
                    )
                )
        backbone.append(nn.BatchNorm2d(self.recover_channels))
        backbone.append(nn.ReLU(inplace=True))
        
        self.backbone = nn.Sequential(*backbone)
        final_layer = []
        ## padding
        if not (self.recover_width - feat_width) == 0:
            final_layer.append(
                nn.Upsample(size=(self.recover_width, self.recover_width), mode='bilinear', align_corners=False)
            )
        final_layer.append(
            nn.Conv2d(
                in_channels=self.recover_channels,
                out_channels=self.recover_channels,
                kernel_size= 3, 
                stride=1,
                padding= 1,
                bias=False
            )
        )
        self.final_layer = nn.Sequential(*final_layer)

    def forward(self, sample):
        x = self.backbone(sample)
        x = self.final_layer(x)
        return x