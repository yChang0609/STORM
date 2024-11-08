import torch
from world_models.modules.JEPA import vision_transformer as vit
from world_models.utils.tensors import trunc_normal_


# -- JEPA model
def init_jepa_encoder(
        patch_size=16,
        model_name='vit_base',
        crop_size=224,
        conv_channels = [],
        conv_strides = [],
        in_chans=3
    )->vit.VisionTransformer:
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size,
        conv_channels = conv_channels,
        conv_strides = conv_strides,
        in_chans=in_chans)
    
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
    for m in encoder.modules():
        init_weights(m)

    return encoder

def load_encoder(
        r_path,
        encoder:vit.VisionTransformer,
        frozen=True
    )->vit.VisionTransformer:
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']
        
        # -- loading encoder
        pretrained_dict = checkpoint['target_encoder']
        for k, v in pretrained_dict.items():
            encoder.state_dict()[k[len("module."):]].copy_(v)

        if frozen:
            for param in encoder.parameters():
                param.requires_grad = False

        print(f'loaded pretrained encoder from epoch {epoch}')
        print(f'jepa model from read-path: {r_path}')
        del checkpoint

    except Exception as e:
        print(f'Encountered exception when loading checkpoint:{e}')
        epoch = 0

    return encoder