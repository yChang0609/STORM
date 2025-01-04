import importlib

vae_configs = {
    "categorical": {
        "vae_class": "world_models.modules.VAE.categorical_vae.CategoricalVAE",
        "dist_head": "world_models.modules.VAE.categorical_vae.CategoricalDistHead",
    },
    "continuous": {
        "vae_class": "world_models.modules.VAE.continuous_vae.ContinuousVAE",
        "dist_head": "world_models.modules.VAE.continuous_vae.GaussianDistHead",
    },
}

coder_configs = {
    "STORM": {
        "network": "world_models.modules.VAE.encoder_decoder.STORM",
    },
    "Dreamer": {
        "network": "world_models.modules.VAE.encoder_decoder.Dreamer",
    },
}

def dynamic_import(module_class_str):
    module_name, class_name = module_class_str.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)