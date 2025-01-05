import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

class STORM:
    class Encoder(nn.Module):
        def __init__(self, 
                     in_channels, in_feature_width ,
                     final_feature_width, 
                     stem_channels=256, num_repeat=2,
                     act='relu', norm='batch'):
            super().__init__()
            self.num_repeat = num_repeat
            self.in_feature_width = int(in_feature_width)
            self.act_fn = nn.ReLU(inplace=True) if act=='relu' else nn.SiLU() #nn.GELU('none')
            self.norm_fn = (
                lambda channel: 
                    nn.Sequential(
                        Rearrange('B C H W -> B H W C'),
                        nn.RMSNorm(channel),
                        Rearrange('B H W C -> B C H W')
                    ) if norm == 'rms' else nn.BatchNorm2d(channel)
            )
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
            backbone.append(self.norm_fn(channels))
            backbone.append(self.act_fn)

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
                backbone.append(self.norm_fn(channels))
                backbone.append(self.act_fn)

            self.backbone = nn.Sequential(*backbone)

            self.last_channels = channels
            self.final_feature_width = feature_width

        def forward(self, x):
            x = self.backbone(x)
            return x

    class Decoder(nn.Module):
        def __init__(self, 
                     in_dim, 
                     last_channels, final_feature_width, encoder_feature_width, 
                     recover_channels, recover_width, 
                     stem_channels=256, num_repeat=2,
                     act='relu', norm='batch'):
            super().__init__()
            backbone = []
            self.recover_channels = int(recover_channels)
            self.recover_width = int(recover_width)
            self.act_fn = nn.ReLU(inplace=True) if act=='relu' else nn.SiLU() # nn.GELU('none')
            self.norm_fn = (
                lambda channel: 
                    nn.Sequential(
                        Rearrange('B C H W -> B H W C'),
                        nn.RMSNorm(channel),
                        Rearrange('B H W C -> B C H W')
                    ) if norm == 'rms' else nn.BatchNorm2d(channel)
            )

            # stem
            backbone.append(nn.Linear(in_dim, int(last_channels*encoder_feature_width*encoder_feature_width), bias=False))
            backbone.append(Rearrange('B (C H W) -> B C H W', C=last_channels, H=encoder_feature_width, W=encoder_feature_width))
            backbone.append(self.norm_fn(last_channels))
            backbone.append(self.act_fn)

            # residual_layer
            # backbone.append(ResidualStack(last_channels, 1, last_channels//4))
            # layers
            channels = last_channels
            feat_width = encoder_feature_width
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
                backbone.append(self.norm_fn(channels))
                backbone.append(self.act_fn)
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
            backbone.append(self.norm_fn(self.recover_channels))
            backbone.append(self.act_fn)

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

class Dreamer:
    class Encoder(nn.Module):
        def __init__(self, 
                     in_channels, in_feature_width ,
                     depth=64, mults=(2, 3, 4, 4),
                    act='gelu', norm='rms', 
                    kernel=5, strided=False):
            super().__init__()
            self.in_channels = in_channels
            current_height = current_width = in_feature_width
            
            self.depths = [depth * mult for mult in mults]
            self.act_fn = getattr(F, act) if hasattr(F, act) else F.gelu
            self.norm_fn = (
                lambda channel: 
                    nn.Sequential(
                        Rearrange('B C H W -> B H W C'),
                        nn.RMSNorm(channel),
                        Rearrange('B H W C -> B C H W')
                    ) if norm == 'rms' else nn.BatchNorm2d(channel)
            )

            self.kernel = kernel
            self.strided = strided

            input_channels = in_channels
            # Define CNN layers for image inputs
            self.conv_layers = nn.ModuleList()
            for depth in self.depths:
                stride = (2 if strided else 1)
                self.conv_layers.append(
                    nn.Conv2d(input_channels, depth, kernel_size=kernel, stride=stride)
                )
                current_height = (current_height + 2 * 0 - (kernel - 1) - 1) // stride + 1
                current_width = (current_width + 2 * 0 - (kernel - 1) - 1) // stride + 1
                
                self.conv_layers.append(self.norm_fn(depth))
                input_channels = depth

            self.last_channels = self.depths[-1]
            self.final_feature_width = current_height

        def forward(self, x):
            for i, layer in enumerate(self.conv_layers):
                x = layer(x)
                x = self.act_fn(x)
                if i % 2 == 1 and not self.strided:
                    x = F.max_pool2d(x, kernel_size=2)
            return x
        
    class Decoder(nn.Module):
        def __init__(self, 
                    in_dim, 
                    last_channels, encoder_feature_width,
                    recover_channels, recover_width, 
                    depth=64, mults=(2, 3, 4, 4),
                    act='silu', norm='rms', 
                    outscale=1.0, kernel=5, strided=False):
            super().__init__()
            self.in_dim = in_dim

            self.depths = [depth * mult for mult in mults]
            self.recover_channels = recover_channels
            self.recover_width = recover_width

            self.act_fn = getattr(F, act) if hasattr(F, act) else F.silu
            self.norm_fn = (
                lambda channel: 
                    nn.Sequential(
                        Rearrange('B C H W -> B H W C'),
                        nn.RMSNorm(channel),
                        Rearrange('B H W C -> B C H W')
                    ) if norm == 'rms' else nn.BatchNorm2d(channel)
            )
            self.outscale = outscale
            self.kernel = kernel
            self.strided = strided
            current_height = current_width = encoder_feature_width

            # Reshspe to encoder last feature shpae form z dim
            reshape_layer = []
            reshape_layer.append(nn.Linear(in_dim, int(last_channels*encoder_feature_width*encoder_feature_width), bias=False))
            reshape_layer.append(Rearrange('B (C H W) -> B C H W', C=last_channels, H=encoder_feature_width, W=encoder_feature_width))
            reshape_layer.append(self.norm_fn(last_channels))
            self.reshape_layer = nn.Sequential(*reshape_layer)

            # Define CNN layers for image outputs
            padding = 0
            dilation = 1
            output_padding = 0
            self.deconv_layers = nn.ModuleList()
            input_channels = last_channels
            for depth in reversed(self.depths[:-1]):
                stride = (2 if strided else 1)
                self.deconv_layers.append(
                    nn.ConvTranspose2d(input_channels, depth, kernel_size=kernel, stride=stride)
                )
                current_height = (current_height - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
                current_width = (current_width - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
                self.deconv_layers.append(self.norm_fn(depth))
                input_channels = depth

            self.img_out = nn.Conv2d(input_channels, self.recover_channels, kernel_size=kernel)
            current_height = (current_height - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
            current_width = (current_width - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1

            # Padding output feature for fit ground truth
            padding_layer = []
            # Add upsampling if needed
            if current_height != self.recover_width or current_width != self.recover_width:
                padding_layer.append(
                    nn.Upsample(size=(self.recover_width, self.recover_width), mode='bilinear', align_corners=False)
                )
            # Add an extra Conv2d layer for fine-tuning
            padding_layer.append(
                nn.Conv2d(
                    in_channels=self.recover_channels,
                    out_channels=self.recover_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                )
            )
            self.padding_layer = nn.Sequential(*padding_layer)

        def forward(self, x):
            x = self.act_fn(self.reshape_layer(x))
            for layer in self.deconv_layers:
                x = self.act_fn(layer(x))
            x = torch.sigmoid(self.padding_layer(self.img_out(x)) * self.outscale)
            return x