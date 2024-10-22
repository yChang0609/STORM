import torch
import torch.nn as nn
from einops import rearrange

class EmbDecoder(nn.Module):
    def __init__(self, emb_channel, in_width, recon_image_width):
        super().__init__()
        backbone = []
        channels = emb_channel 
        feat_width = 4
        original_in_channels = 3
        self.in_width = int(in_width)
        self.recon_image_width=recon_image_width
        width = in_width
        while True:
            if width == recon_image_width//2: 
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
            backbone.append(nn.BatchNorm2d(channels))
            backbone.append(nn.ReLU(inplace=True))
            width *=2

        backbone.append(
            nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=original_in_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )
        self.backbone = nn.Sequential(*backbone)

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "B (H W) C  -> B C H W",B = batch_size,H=self.in_width)
        obs_hat = self.backbone(x)
        obs_hat = rearrange(obs_hat, "B C H W  -> B 1 C H W",B = batch_size,H=self.recon_image_width)
        return obs_hat
    
    def decode_video(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "B L P C  -> (B L) P C")
        obs_hat = self.forward(x)
        obs_hat = rearrange(obs_hat, "(B L) 1 C H W -> B L C H W",B = batch_size, H=self.recon_image_width)
        return obs_hat

def init_jepa_decoder(
        emb_channel,
        in_width,
        recon_image_width
)->EmbDecoder:
    return EmbDecoder(emb_channel, in_width, recon_image_width)

def load_jepa_decoder(
        r_path,
        decoder:EmbDecoder,
        frozen=True
)->EmbDecoder:
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']
        
        # -- loading encoder
        pretrained_dict = checkpoint['visual_model']
        decoder.load_state_dict(pretrained_dict)

        if frozen:
            for param in decoder.parameters():
                param.requires_grad = False
        print(f'loaded pretrained encoder from epoch {epoch}')
        print(f'jepa model from read-path: {r_path}')
        del checkpoint

    except Exception as e:
        print(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return decoder