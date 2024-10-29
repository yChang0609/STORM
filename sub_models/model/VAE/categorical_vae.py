import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical
from einops import rearrange
from einops.layers.torch import Rearrange
from sub_models.model.VAE.base import *
import sub_models.model.VAE.encoder_decoder as model
from math import sqrt

class DistHead(nn.Module):
    '''
    Dist: abbreviation of distribution
    '''
    def __init__(self, image_feat_dim, stoch_dim) -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.post_head = nn.Linear(image_feat_dim, stoch_dim*stoch_dim)

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        probs = F.softmax(logits, dim=-1)
        mixed_probs = mixing_ratio * torch.ones_like(probs) / self.stoch_dim + (1-mixing_ratio) * probs
        logits = torch.log(mixed_probs)
        return logits

    def forward_post(self, x):
        logits = self.post_head(x)
        logits = rearrange(logits, "B (K C) -> B K C", K=self.stoch_dim)
        logits = self.unimix(logits)
        return logits
   
class CategoricalVAE(BaseVAE):
    '''Categorical Variational Auto Encoder'''
    def __init__(self, stoch_dim, in_channels, in_feature_width, use_amp):
        super().__init__()
        self.use_amp = use_amp
        final_feature_width = 2
        self.in_channels=int(in_channels)
        self.in_feature_width=int(in_feature_width)
        self.stoch_dim = stoch_dim
        self.stoch_flattened_dim = self.stoch_dim*self.stoch_dim

        stem_channels=256
        encoder_in_channels = in_channels
        r = int(sqrt(in_channels//encoder_in_channels))
        
        self.pixel_shuffle = nn.PixelShuffle(r)
        self.pixel_unshuffle = nn.PixelUnshuffle(r)

        self.encoder = model.Encoder(
            in_channels=encoder_in_channels,
            in_feature_width=in_feature_width*r, 
            final_feature_width=final_feature_width,
            stem_channels=stem_channels,
            num_repeat=0
        )

        self.dist_head = DistHead(
            image_feat_dim=self.encoder.last_channels*self.encoder.final_feature_width*self.encoder.final_feature_width,
            stoch_dim=self.stoch_dim
        )

        self.decoder = model.Decoder(
            in_dim=stoch_dim*stoch_dim, 
            last_channels=self.encoder.last_channels, 
            final_feature_width=self.encoder.final_feature_width, 
            recover_channels=encoder_in_channels,
            recover_width=in_feature_width*r,
            stem_channels=stem_channels,
            num_repeat=self.encoder.num_repeat
        )

    def stright_throught_gradient(self, logits, sample_mode="random_sample"):
        dist = OneHotCategorical(logits=logits)
        if sample_mode == "random_sample":
            sample = dist.sample() + dist.probs - dist.probs.detach()
        elif sample_mode == "mode":
            sample = dist.mode
        elif sample_mode == "probs":
            sample = dist.probs
        return sample
    
    def flatten_sample(self, sample):
        return rearrange(sample, "B K C -> B (K C)")

    # -- VAE interface
    def encode(self, input: Tensor) -> List[Tensor]:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            x = rearrange(input, "B (H W) C -> B C H W", C=self.in_channels, H=self.in_feature_width)
            x = self.pixel_shuffle(x)
            x = self.encoder(x)
            x = rearrange(x, "B C H W  -> B (C H W)", C=self.encoder.last_channels, H=self.encoder.final_feature_width)
            post_logits = self.dist_head.forward_post(x)
        return [post_logits]
    
    def decode(self, z: Tensor) -> Tensor:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            z = self.flatten_sample(z)
            x = self.decoder(z)
            x = self.pixel_unshuffle(x)
            x = rearrange(x, "B C H W  -> B (H W) C", C=self.in_channels, H=self.in_feature_width)
        return x
    
    def sample(self, params:List[Tensor], **kwargs) -> Tensor:
        return self.stright_throught_gradient(params[0], sample_mode="random_sample")
    
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        post_logits = self.encode(input)[0]
        z = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
        return  [self.decode(z), input, post_logits]
