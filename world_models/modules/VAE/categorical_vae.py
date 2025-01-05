import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical
from einops import rearrange
from einops.layers.torch import Rearrange
from world_models.modules.VAE.vae_base import *


class CategoricalDistHead(BaseDistHead):
    '''
    Dist: abbreviation of distribution
    '''
    def __init__(self, feat_dim, stoch_dim) -> None:
        super().__init__(feat_dim, stoch_dim)
        self.post_head = nn.Linear(feat_dim, stoch_dim*stoch_dim)

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        probs = F.softmax(logits, dim=-1)
        mixed_probs = mixing_ratio * torch.ones_like(probs) / self.stoch_dim + (1-mixing_ratio) * probs
        logits = torch.log(mixed_probs)
        return logits

    def forward(self, x):
        logits = self.post_head(x)
        logits = rearrange(logits, "B (K C) -> B K C", K=self.stoch_dim)
        logits = self.unimix(logits)
        return CategoricalDistributionParams([logits])
   
class CategoricalVAE(BaseVAE):
    '''Categorical Variational Auto Encoder'''
    def __init__(self, 
                 z_dim:int, 
                 in_channels:int, in_feature_width:int, use_amp, 
                 coder_type, coder_params, pixel_suffle_channels=None,
                 **kwargs):
        
        self.use_amp = use_amp
        self.in_channels=int(in_channels)
        self.in_feature_width=int(in_feature_width)
        self.stoch_dim = z_dim
        self.stoch_flattened_dim = self.stoch_dim*self.stoch_dim
        super().__init__(
            stoch_flattened_dim=self.stoch_flattened_dim,
            in_channels=in_channels, in_feature_width=in_feature_width,
            pixel_suffle_channels=pixel_suffle_channels,
            coder_type=coder_type,
            coder_params=coder_params,
            **kwargs,
        )

        self.dist_head = CategoricalDistHead(
            feat_dim=self.encoder.last_channels*self.encoder.final_feature_width*self.encoder.final_feature_width,
            stoch_dim=self.stoch_dim
        )

    def stright_throught_gradient(self, params:CategoricalDistributionParams, sample_mode="random_sample"):
        dist = OneHotCategorical(logits=params.logits())
        if sample_mode == "random_sample":
            sample = dist.sample() + dist.probs - dist.probs.detach()
        elif sample_mode == "mode":
            sample = dist.mode
        elif sample_mode == "probs":
            sample = dist.probs
        return sample
    

    # -- VAE interface
    def encode(self, input: Tensor) -> CategoricalDistributionParams:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # x = rearrange(input, "B (H W) C -> B C H W", C=self.in_channels, H=self.in_feature_width)
            x = self.pixel_shuffle(input)
            x = self.encoder(x)
            x = rearrange(x, "B C H W  -> B (C H W)", C=self.encoder.last_channels, H=self.encoder.final_feature_width)
            return self.dist_head(x)
        
    
    def decode(self, z: Tensor) -> Tensor:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            z = self.flatten_sample(z)
            x = self.decoder(z)
            x = self.pixel_unshuffle(x)
            # x = rearrange(x, "B C H W  -> B (H W) C", C=self.in_channels, H=self.in_feature_width)
        return x
    
    def sample(self, params:CategoricalDistributionParams, **kwargs) -> Tensor:
        sample_mode = kwargs.get('sample_mode', "random_sample")
        return self.stright_throught_gradient(params, sample_mode=sample_mode)
    
    def flatten_sample(self, sample):
        return rearrange(sample, "B K C -> B (K C)")
    
    def forward(self, input: Tensor, **kwargs) -> List[Tensor|CategoricalDistributionParams]:
        post_logits = self.encode(input)
        sample_mode = kwargs.get('sample_mode', "random_sample")
        z = self.sample(post_logits, sample_mode=sample_mode)
        return  [self.decode(z), input, post_logits]
