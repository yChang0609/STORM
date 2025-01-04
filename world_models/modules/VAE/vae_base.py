from typing import List, Any, TypeVar
# from torch import tensor as Tensor


from torch import nn
from abc import abstractmethod
from math import sqrt

from world_models.utils.utils import coder_configs, dynamic_import
Tensor = TypeVar('torch.tensor')


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

        ### STORM 
        ## encoder
        # final_feature_width, stem_channels=256, num_repeat=2):

        ## decoer
        # final_feature_width, stem_channels=256, num_repeat=2):
        
        #### Dreamer
        ## encoder
        # depth=64, mults=(2, 3, 4, 4), layers=3, 
        # act='gelu', norm='rms', 
        # symlog=True, kernel=5, strided=False):

        ## decoder
        # depth=64, mults=(2, 3, 4, 4), layers=3,
        # act='gelu', norm='rms', 
        # outscale=1.0, kernel=5, strided=False):
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

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, param:List[Tensor], **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass
