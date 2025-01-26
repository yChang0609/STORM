import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
# import torch.utils.checkpoint as checkpoint

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
            self.act_fn = nn.ReLU(inplace=True) if act=='relu' else nn.SiLU(inplace=True) #nn.GELU('none')
            self.norm_fn = lambda shape:nn.RMSNorm(shape) if norm == 'rms' else nn.BatchNorm2d(shape[-3])
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
            backbone.append(self.norm_fn((channels, feature_width, feature_width)))
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
                backbone.append(self.norm_fn((channels, feature_width, feature_width)))
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
            self.act_fn = nn.ReLU(inplace=True) if act=='relu' else nn.SiLU(inplace=True) # nn.GELU('none')
            self.norm_fn = lambda shape:nn.RMSNorm(shape) if norm == 'rms' else nn.BatchNorm2d(shape[-3])

            # stem
            backbone.append(nn.Linear(in_dim, int(last_channels*encoder_feature_width*encoder_feature_width), bias=False))
            backbone.append(Rearrange('B (C H W) -> B C H W', C=last_channels, H=encoder_feature_width, W=encoder_feature_width))
            backbone.append(self.norm_fn((last_channels, encoder_feature_width, encoder_feature_width)))
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
                backbone.append(self.norm_fn((channels, feat_width, feat_width)))
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
            backbone.append(self.norm_fn((self.recover_channels, feat_width, feat_width)))
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
            self.act_fn = nn.ReLU(inplace=True) if act=='relu' else nn.SiLU(inplace=True) #nn.GELU('none')
            self.norm_fn = lambda shape:nn.RMSNorm(shape) if norm == 'rms' else nn.BatchNorm2d(shape[-3])

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
                
                self.conv_layers.append(self.norm_fn((depth, current_height, current_width)))
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

            self.act_fn = nn.ReLU(inplace=True) if act=='relu' else nn.SiLU(inplace=True) #nn.GELU('none')
            self.norm_fn = lambda shape:nn.RMSNorm(shape) if norm == 'rms' else nn.BatchNorm2d(shape[-3])
            self.outscale = outscale
            self.kernel = kernel
            self.strided = strided
            current_height = current_width = encoder_feature_width

            # Reshspe to encoder last feature shpae form z dim
            reshape_layer = []
            reshape_layer.append(nn.Linear(in_dim, int(last_channels*encoder_feature_width*encoder_feature_width), bias=False))
            reshape_layer.append(Rearrange('B (C H W) -> B C H W', C=last_channels, H=encoder_feature_width, W=encoder_feature_width))
            reshape_layer.append(self.norm_fn((last_channels, encoder_feature_width, encoder_feature_width)))
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
                self.deconv_layers.append(self.norm_fn((depth, current_height, current_width)))
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


if __name__ == '__main__':
    in_channels = 3
    in_feature_width = 224
    device = torch.device('cuda:0')

    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim as optim
    import time
    # Mock dataset creation (e.g., random image-like data)
    def generate_mock_data(num_samples, in_channels, in_feature_width):
        data = torch.rand(num_samples, in_channels, in_feature_width, in_feature_width)
        return data

    # Hyperparameters
    num_samples = 1000
    batch_size = 16 * 16
    epochs = 10
    learning_rate = 1e-3

    # Generate mock dataset
    data = generate_mock_data(num_samples, in_channels, in_feature_width)
    dataset = TensorDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    encoder = Dreamer.Encoder(
        in_channels, in_feature_width ,
        depth=64, mults=(2, 3, 4, 4),
        act='silu', norm='rms', 
        kernel=5, strided=True
    ).to(device)

    head = nn.Linear(
        encoder.last_channels*encoder.final_feature_width*encoder.final_feature_width, 
        32*32
    ).to(device)

    decoder = Dreamer.Decoder(
        32*32, 
        encoder.last_channels, encoder.final_feature_width,
        in_channels, in_feature_width, 
        depth=8, mults=(2, 3, 4, 4),
        act='silu', norm='rms', 
        outscale=1.0, kernel=5, strided=True
    ).to(device)

    # Generate mock dataset
    data = generate_mock_data(num_samples, in_channels, in_feature_width)
    dataset = TensorDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define optimizer and loss function
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(head.parameters()))

    print(encoder)
    print(head)
    print(decoder)

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (inputs,) in enumerate(data_loader):
            # Move data to device
            inputs = inputs.to(device)

            # Forward pass
            encoded = encoder(inputs)
            latent = head(encoded.view(encoded.size(0), -1))
            # latent = latent.view(latent.size(0), 32, 32)
            reconstructed = decoder(latent)

            # Compute loss
            loss = criterion(reconstructed, inputs)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Monitor GPU usage
            if batch_idx % 10 == 0:
                free, total = torch.cuda.mem_get_info(device)
                mem_used_MB = (total - free) / 1024 ** 2
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(data_loader)}], GPU Memory Used: {mem_used_MB:.2f} MB")

        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(data_loader):.4f}, Time: {epoch_time:.2f}s")

    print("Training complete.")
