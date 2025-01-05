import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from world_models.modules.VAE.vae_base import *
import world_models.modules.VAE.encoder_decoder as model
from math import sqrt

class GaussianDistHead(BaseDistHead):
    def __init__(self, feat_dim, stoch_dim):
        super().__init__(feat_dim, stoch_dim)
        self.mu_mlp = nn.Linear(feat_dim, stoch_dim)
        self.logvar_mlp = nn.Linear(feat_dim, stoch_dim)

    def forward(self, x):
        return GaussianDistributionParams([self.mu_mlp(x), self.logvar_mlp(x)])

class ContinuousVAE(BaseVAE):
    def __init__(self, 
                 z_dim:int, 
                 in_channels:int, in_feature_width:int, use_amp, 
                 coder_type, coder_params, pixel_suffle_channels=None,
                 **kwargs):
        self.use_amp = use_amp
        self.in_channels=int(in_channels)
        self.in_feature_width=int(in_feature_width)
        self.stoch_dim = z_dim
        self.stoch_flattened_dim = z_dim
        super().__init__(
            stoch_flattened_dim=self.stoch_flattened_dim,
            in_channels=in_channels, in_feature_width=in_feature_width,
            pixel_suffle_channels=pixel_suffle_channels,
            coder_type=coder_type,
            coder_params=coder_params,
            **kwargs,
        )

        self.dist_head = GaussianDistHead(
            feat_dim=self.encoder.last_channels*self.encoder.final_feature_width*self.encoder.final_feature_width,
            stoch_dim=self.stoch_dim
        )
        
    def reparameterize(self, params:GaussianDistributionParams) -> Tensor:
        std = torch.exp(0.5 * params.logvar())
        eps = torch.randn_like(std)
        return eps * std + params.mu()
    
    # -- VAE interface
    def encode(self, input: Tensor) -> GaussianDistributionParams:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # x = rearrange(input, "B (H W) C -> B C H W", C=self.in_channels, H=self.in_feature_width)
            x = self.pixel_shuffle(input)
            x = self.encoder(x)
            x = rearrange(x, "B C H W  -> B (C H W)", C=self.encoder.last_channels, H=self.encoder.final_feature_width)
            return self.dist_head(x)
        
    
    def decode(self, z: Tensor) -> Tensor:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            x = self.decoder(z)
            x = self.pixel_unshuffle(x)
            # x = rearrange(x, "B C H W  -> B (H W) C", C=self.in_channels, H=self.in_feature_width)
        return x
    
    def sample(self, params:GaussianDistributionParams, **kwargs) -> Tensor:
        return self.reparameterize(params)
    
    def flatten_sample(self, sample):
        return sample # nothing needs to change
    
    def forward(self, input: Tensor, **kwargs) -> List[Tensor|GaussianDistributionParams]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]