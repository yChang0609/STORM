from typing import List, Any, TypeVar
# from torch import tensor as Tensor


from torch import nn
from abc import abstractmethod
from math import sqrt
from einops import rearrange

from world_models.utils.utils import coder_configs, dynamic_import
Tensor = TypeVar('torch.tensor')

class BaseDistributionParams():
    def __init__(self, params):
        self.params = params
    def detach(self):
        raise NotImplementedError
    def batch_rearrange(self, pattern: str):
        raise NotImplementedError
    def __getitem__(self, idx):
        raise NotImplementedError
    
class CategoricalDistributionParams(BaseDistributionParams):
    def __init__(self, params:List[Tensor]):
        super().__init__(params)
        self.shape = "K C"
    def detach(self):
        return CategoricalDistributionParams([self.params[0].detach()])
    def batch_rearrange(self, org_pattern: str, to_pattern: str, **axes_lengths):
        self.params[0] = rearrange(self.params[0], f"{org_pattern} {self.shape} -> {to_pattern} {self.shape}", **axes_lengths)
    def __getitem__(self, idx):
        return CategoricalDistributionParams([self.params[0][idx]])
    def logits(self):
        return self.params[0]

class GaussianDistributionParams(BaseDistributionParams):
    def __init__(self, params:List[Tensor]):
        super().__init__(params)
        self.shape = "Z"
    def detach(self):
        return GaussianDistributionParams([self.params[0].detach(), self.params[1].detach()])

    def batch_rearrange(self, org_pattern: str, to_pattern: str, **axes_lengths):
        self.params[0] = rearrange(self.params[0], f"{org_pattern} {self.shape} -> {to_pattern} {self.shape}", **axes_lengths)
        self.params[1] = rearrange(self.params[1], f"{org_pattern} {self.shape} -> {to_pattern} {self.shape}", **axes_lengths)

    def __getitem__(self, idx):
        return GaussianDistributionParams([self.params[0][idx], self.params[1][idx]])
    
    def mu(self):
        return self.params[0]
    def logvar(self):
        return self.params[1]
    
class BaseDistHead(nn.Module):
    def __init__(self, feat_dim, stoch_dim) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.stoch_dim = stoch_dim

class BaseVAE(nn.Module):
    def __init__(self,
                 stoch_flattened_dim,     
                 in_channels:int, in_feature_width:int, pixel_suffle_channels=None,
                 coder_type="STORM",
                 coder_params=None,
                 **kwargs) -> None:
                #  in_channels:int, in_feature_width:int, 
                #  stem_channels:int, stem_repeat:int,
                #  final_feature_width:int, pixel_suffle_channels=None) -> None:
        super(BaseVAE, self).__init__()
        
        encoder_in_channels = in_channels if pixel_suffle_channels==None else pixel_suffle_channels
        r = int(sqrt(in_channels//encoder_in_channels))
        self.pixel_shuffle = nn.PixelShuffle(r)
        self.pixel_unshuffle = nn.PixelUnshuffle(r)

        if coder_type not in coder_configs:
            raise ValueError(f"Unsupported VAE type: {coder_type}")
        config = coder_configs[coder_type]
        network = dynamic_import(config["network"])

        self.encoder = network.Encoder(
            in_channels=encoder_in_channels,
            in_feature_width=in_feature_width*r, 
            **coder_params,
        )
        self.decoder = network.Decoder(
            in_dim=stoch_flattened_dim, 
            last_channels=self.encoder.last_channels, encoder_feature_width=self.encoder.final_feature_width, 
            recover_channels=encoder_in_channels, recover_width=in_feature_width*r,
            **coder_params,
        )

    def encode(self, input: Tensor) -> BaseDistributionParams:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, param:BaseDistributionParams, **kwargs) -> Tensor:
        raise NotImplementedError
    
    def flatten_sample(self, sample) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass
