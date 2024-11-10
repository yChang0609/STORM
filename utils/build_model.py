from agents import agents

def build_world_model(params, action_dims):
    wm_type = params["Models"]["WorldModel"]["ModleName"] 
    if wm_type == "JEPA_WM":
        from world_models.jepa_world_models import JEPAWorldModel
        wm = JEPAWorldModel(
            # Input setting
            action_dims=action_dims,
            in_channels=params["Models"]["WorldModel"]["InChannels"],
            in_width=params["BasicSettings"]["ImageSize"],

            # JEPA
            patch_size=params["Models"]["WorldModel"]["JEPAParams"]["PatchSize"],
            jepa_size=params["Models"]["WorldModel"]["JEPAParams"]["ModelSize"],
            jepa_load_path=params["Models"]["WorldModel"]["JEPAParams"]["ModelPath"],
            
            # VAE
            vae_type=params["Models"]["WorldModel"]["VAEParams"]["Type"], 
            stoch_dim=params["Models"]["WorldModel"]["VAEParams"]["StochasticDim"], 
            final_feature_width=params["Models"]["WorldModel"]["VAEParams"]["EncodeFinalFeatureWidth"], 
            stem_channels=params["Models"]["WorldModel"]["VAEParams"]["EncodeStemChannels"], 
            stem_repeat=params["Models"]["WorldModel"]["VAEParams"]["EncodeStemRepeatNum"], 
            
            # Transformer
            transformer_max_length=params["Models"]["WorldModel"]["TransformerParams"]["MaxLength"],
            transformer_hidden_dim=params["Models"]["WorldModel"]["TransformerParams"]["HiddenDim"],
            transformer_num_layers=params["Models"]["WorldModel"]["TransformerParams"]["NumLayers"],
            transformer_num_heads=params["Models"]["WorldModel"]["TransformerParams"]["NumHeads"],

            use_amp=params["Models"]["use_amp"]
        )

    elif wm_type == "STORM":
        from world_models.storm_world_models import STORMWorldModel 
        wm = STORMWorldModel(
            # Input setting
            action_dims=action_dims,
            in_channels=params["Models"]["WorldModel"]["InChannels"],
            in_width=params["BasicSettings"]["ImageSize"],

            # VAE
            vae_type=params["Models"]["WorldModel"]["VAEParams"]["Type"], 
            stoch_dim=params["Models"]["WorldModel"]["VAEParams"]["StochasticDim"], 
            final_feature_width=params["Models"]["WorldModel"]["VAEParams"]["EncodeFinalFeatureWidth"], 
            stem_channels=params["Models"]["WorldModel"]["VAEParams"]["EncodeStemChannels"], 
            stem_repeat=params["Models"]["WorldModel"]["VAEParams"]["EncodeStemRepeatNum"], 
            
            # Transformer
            transformer_max_length=params["Models"]["WorldModel"]["TransformerParams"]["MaxLength"],
            transformer_hidden_dim=params["Models"]["WorldModel"]["TransformerParams"]["HiddenDim"],
            transformer_num_layers=params["Models"]["WorldModel"]["TransformerParams"]["NumLayers"],
            transformer_num_heads=params["Models"]["WorldModel"]["TransformerParams"]["NumHeads"],
            use_amp=params["Models"]["use_amp"]
        )
    return wm.cuda()

def build_agent(params, action_dim):
    return agents.ActorCriticAgent(
        feat_dim= sum(params["Models"]["Agent"]["InputFeature"]),
        num_layers=params["Models"]["Agent"]["NumLayers"],
        hidden_dim=params["Models"]["Agent"]["HiddenDim"],
        action_dim=action_dim,
        gamma=float(params["Models"]["Agent"]["Gamma"]),
        lambd=float(params["Models"]["Agent"]["Lambda"]),
        entropy_coef=float(params["Models"]["Agent"]["EntropyCoef"]),
    ).cuda()