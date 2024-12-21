import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange 
from einops.layers.torch import Rearrange

# World model basic class
from world_models.world_model_base import WorldModelBase

# VAE

# JEPA
from torchvision import transforms
from world_models.modules.JEPA.jepa_encoder import init_jepa_encoder, load_encoder
from world_models.modules.JEPA.jepa_decoder import init_jepa_decoder, load_decoder


# Dynamic model transformer
from world_models.modules.Transformer.attention_blocks import get_subsequent_mask_with_batch_length, get_subsequent_mask
from world_models.modules.Transformer.transformer_model import StochasticTransformerKVCache

# Predition
from world_models.modules.Predictior.prediction_decoders import RewardDecoder, TerminationDecoder

# Funciton
from world_models.utils.action2onehot import actions2onehot
from world_models.utils.functions_losses import SymLogTwoHotLoss, MSELoss, SymLogLoss, CategoricalKLDivLossWithFreeBits, symexp
from world_models.utils.logging import error_msg

'''
Import Agent for interaction with the world model, which processes and responds to image data
'''
from agents.agents import ActorCriticAgent

from math import sqrt

def tensor_unormalize(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return tensor * torch.tensor(std).view(3, 1, 1).cuda() + torch.tensor(mean).view(3, 1, 1).cuda()


class JEPAWorldModel(WorldModelBase):
    def __init__(self, 
                 action_dims,
                 in_channels, in_width,
                 patch_size, jepa_size, jepa_load_path:tuple,
                 vae_type, stoch_dim, stem_channels, stem_repeat, final_feature_width,
                 transformer_max_length, transformer_hidden_dim, transformer_num_layers, transformer_num_heads,
                 symlog, use_amp):

        super().__init__()
        self.action_dims = action_dims
        self.transformer_hidden_dim = transformer_hidden_dim
        self.stoch_dim = stoch_dim
        self.use_amp = use_amp
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.symlog = symlog

        # JEPA model 
        jepa_encoder = init_jepa_encoder(
            patch_size=patch_size,
            model_name=jepa_size,
            crop_size=in_width,
            in_chans=in_channels
        )
        jepa_encoder = load_encoder(
            r_path=jepa_load_path[0],
            encoder=jepa_encoder,
            frozen=True
        )
        jepa_feat_width = int(sqrt(jepa_encoder.patch_embed.num_patches))
        self.jepa_encoder = nn.Sequential(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            jepa_encoder,
            Rearrange('B (H W) C -> B C H W',H=jepa_feat_width),
            )
        jepa_decoder = init_jepa_decoder(
            emb_channel=jepa_encoder.embed_dim,
            in_width=jepa_feat_width,
            recon_image_width=in_width
        )
        jepa_decoder = load_decoder(
            r_path=jepa_load_path[1],
            decoder=jepa_decoder
        )
        self.jepa_decoder = jepa_decoder

        # VAE
        if vae_type == "categorical":
            from world_models.modules.VAE.categorical_vae import CategoricalVAE as vae
            from world_models.modules.VAE.categorical_vae import CategoricalDistHead as DistHead
            self.stoch_flattened_dim = stoch_dim*stoch_dim

        elif vae_type == "continuous":
            from world_models.modules.VAE.continuous_vae import ContinuousVAE as vae
            from world_models.modules.VAE.continuous_vae import GaussianDistHead as DistHead
            self.stoch_flattened_dim = stoch_dim
            assert error_msg(f"Continuous VAE Loss not design.")

        else:
            assert error_msg(f"This VAE type not implement{vae_type}.")

        self._vae = vae(
            z_dim=stoch_dim,
            in_channels=jepa_encoder.embed_dim, 
            in_feature_width=jepa_feat_width,
            stem_channels=stem_channels, 
            stem_repeat=stem_repeat,
            final_feature_width=final_feature_width, 
            use_amp=use_amp,
        )
        # TODO : refactor VAE create
        # import importlib
        # def dynamic_import(module_class_str):
        #     module_name, class_name = module_class_str.rsplit(".", 1)
        #     module = importlib.import_module(module_name)
        #     return getattr(module, class_name)?
        # vae_configs = {
        #     "categorical": {
        #         "vae_class": "world_models.modules.VAE.categorical_vae.CategoricalVAE",
        #         "dist_head": "world_models.modules.VAE.categorical_vae.CategoricalDistHead",
        #         "stoch_flattened_dim": lambda stoch_dim: stoch_dim * stoch_dim,
        #     },
        #     "continuous": {
        #         "vae_class": "world_models.modules.VAE.continuous_vae.ContinuousVAE",
        #         "dist_head": "world_models.modules.VAE.continuous_vae.GaussianDistHead",
        #         "stoch_flattened_dim": lambda stoch_dim: stoch_dim,
        #     },
        # }

        # # 確保提供的 VAE 類型是支持的
        # if vae_type not in vae_configs:
        #     raise ValueError(f"Unsupported VAE type: {vae_type}")

        # # 獲取對應配置
        # config = vae_configs[vae_type]

        # # 動態導入 VAE 類和分布頭
        # vae = dynamic_import(config["vae_class"])
        # DistHead = dynamic_import(config["dist_head"])
        # self.stoch_flattened_dim = config["stoch_flattened_dim"](stoch_dim)

        # # 檢查特殊條件
        # if vae_type == "continuous":
        #     raise NotImplementedError("Continuous VAE Loss not designed.")

        # # 初始化 VAE
        # self._vae = vae(
        #     z_dim=stoch_dim,
        #     in_channels=jepa_encoder.embed_dim,
        #     in_feature_width=jepa_feat_width,
        #     stem_channels=stem_channels,
        #     stem_repeat=stem_repeat,
        #     final_feature_width=final_feature_width,
        #     use_amp=use_amp,
        # )
        
        # Transformer
        self.storm_transformer = StochasticTransformerKVCache(
            stoch_dim=self.stoch_flattened_dim,
            action_dim=sum(action_dims),
            feat_dim=transformer_hidden_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            max_length=transformer_max_length,
            dropout=0.1
        )
        self._prior_dist_head = DistHead(            
            feat_dim=transformer_hidden_dim,
            stoch_dim=stoch_dim
        )

        # Predictor
        self.reward_decoder = RewardDecoder(
            num_classes=255,
            transformer_hidden_dim=transformer_hidden_dim
        )
        self.termination_decoder = TerminationDecoder(
            transformer_hidden_dim=transformer_hidden_dim
        )
        
        self.mse_loss_func = SymLogLoss() if symlog else MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def encode_obs(self, obs):
        batch_size, batch_length = obs.shape[:2]
        obs = rearrange(obs, "B L C H W -> (B L) C H W") # process input shape
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            with torch.no_grad():
                emb = self.jepa_encoder(obs) # [Batch&Length Channels sqrt(patch) sqrt(patch)]
            post_logits = self._vae.encode(emb)
            sample = self._vae.sample(post_logits, sample_mode="random_sample")
            flattened_sample = self._vae.flatten_sample(sample)
        # emb = rearrange(emb, "(B L) C H W -> B L C H W",B=batch_size) # process output shape
        emb = rearrange(emb, "(B L) C H W -> B L C H W",B=batch_size) # process output shape
        flattened_sample = rearrange(sample, "(B L) K C -> B L (K C)",B=batch_size) # process output shape
        return flattened_sample, emb
    
    # calculate last distribution feature
    def calc_last_dist_feat(self, latent, actions):
        batch_size, batch_length = latent.shape[:2]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask(latent)
            dist_feat = self.storm_transformer(latent, actions2onehot(actions, self.action_dims), temporal_mask)
            last_dist_feat = dist_feat[:, -1:]

            _last_dist_feat = rearrange(last_dist_feat, "B L C -> (B L) C") 
            prior_logits = self._prior_dist_head(_last_dist_feat)
            prior_sample = self._vae.sample([prior_logits], sample_mode="random_sample")
            prior_flattened_sample = self._vae.flatten_sample(prior_sample)
            prior_flattened_sample = rearrange(prior_flattened_sample, "(B L) C -> B L C",B=batch_size) 

        return prior_flattened_sample, last_dist_feat
    
    def predict_next(self, last_flattened_sample, actions, log_video=True):
        batch_size, batch_length = last_flattened_sample.shape[:2]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            dist_feat = self.storm_transformer.forward_with_kv_cache(last_flattened_sample, actions2onehot(actions, self.action_dims))

            _dist_feat = rearrange(dist_feat, "B L C -> (B L) C") 
            prior_logits = self._prior_dist_head(_dist_feat)

            # decoding
            prior_sample = self._vae.sample([prior_logits], sample_mode="random_sample")
            prior_flattened_sample = self._vae.flatten_sample(prior_sample)
            prior_flattened_sample = rearrange(prior_flattened_sample, "(B L) C -> B L C",B=batch_size) 

            if log_video:
                emb_hat = self._vae.decode(prior_sample)
                emb_hat = symexp(emb_hat) if self.symlog else emb_hat
                obs_hat = self.jepa_decoder(emb_hat)
                obs_hat = rearrange(obs_hat, "(B L) C H W -> B L C H W",B=batch_size) 
            else:
                obs_hat = None

    
            reward_hat = self.reward_decoder(dist_feat)
            reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
            termination_hat = self.termination_decoder(dist_feat)
            termination_hat = termination_hat > 0

        return obs_hat, reward_hat, termination_hat, prior_flattened_sample, dist_feat
    

    def imagine_data(self, 
                     agent: ActorCriticAgent, 
                     sample_obs, sample_action,
                     imagine_batch_size, imagine_batch_length, 
                     log_video, logger):
        self.init_imagine_buffer(self.stoch_flattened_dim, self.transformer_hidden_dim, self.action_dims,
                            imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype)
        obs_hat_list = []

        self.storm_transformer.reset_kv_cache_list(imagine_batch_size, dtype=self.tensor_dtype)
        # context
        context_latent, embedding = self.encode_obs(sample_obs)
        for i in range(sample_obs.shape[1]):  # context_length is sample_obs.shape[1]
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                context_latent[:, i:i+1],
                sample_action[:, i:i+1],
                log_video=log_video
            )
        self.latent_buffer[:, 0:1] = last_latent
        self.hidden_buffer[:, 0:1] = last_dist_feat

        # imagine
        for i in range(imagine_batch_length):
            action = agent.sample(torch.cat([self.latent_buffer[:, i:i+1], self.hidden_buffer[:, i:i+1]], dim=-1))
            self.action_buffer[:, i:i+1] = action

            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                self.latent_buffer[:, i:i+1], self.action_buffer[:, i:i+1], log_video=log_video)

            self.latent_buffer[:, i+1:i+2] = last_latent
            self.hidden_buffer[:, i+1:i+2] = last_dist_feat
            self.reward_hat_buffer[:, i:i+1] = last_reward_hat
            self.termination_hat_buffer[:, i:i+1] = last_termination_hat
            if log_video:
                obs_hat_list.append(last_obs_hat[::imagine_batch_size//16])  # uniform sample vec_env

        if log_video:
            logger.log("Imagine/sample_video", torch.clamp(sample_obs[::imagine_batch_size//16], 0, 1).cpu().float().detach().numpy())
            logger.log("Imagine/jepa_rec_video", torch.clamp(tensor_unormalize(self.jepa_decoder.decode_video(embedding[::imagine_batch_size//16])), 0, 1).cpu().float().detach().numpy())
            logger.log("Imagine/predict_video", torch.clamp(tensor_unormalize(torch.cat(obs_hat_list, dim=1)), 0, 1).cpu().float().detach().numpy())

        return torch.cat([self.latent_buffer, self.hidden_buffer], dim=-1), self.action_buffer, self.reward_hat_buffer, self.termination_hat_buffer
        
    def update(self, obs, actions, reward, termination, logger=None, log_video=False):
        self.train()
        batch_size, batch_length = obs.shape[:2]
        # reshape [B L * ]-> [B *]
        vae_obs = rearrange(obs, "B L C H W -> (B L) C H W")
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # encoding
            with torch.no_grad():
                emb = self.jepa_encoder(vae_obs) # [Batch&Length Channels sqrt(patch) sqrt(patch)]
            post_logits = self._vae.encode(emb)
            sample = self._vae.sample(post_logits, sample_mode="random_sample")
            flattened_sample = self._vae.flatten_sample(sample)
            
            # decoding image
            emb_hat = self._vae.decode(sample)

            # reshape [B * ]-> [B L *]
            post_logits = rearrange(post_logits[0], "(B L) K C -> B L K C",B=batch_size)
            flattened_sample = rearrange(sample, "(B L) K C -> B L (K C)",B=batch_size) # process output shape
            # emb = rearrange(emb, "(B L) C H W -> B L C H W",B=batch_size)
            emb = rearrange(emb, "(B L) C H W -> B L C H W",B=batch_size)
            emb_hat = rearrange(emb_hat, "(B L) C H W -> B L C H W",B=batch_size)

            # transformer
            temporal_mask = get_subsequent_mask_with_batch_length(batch_length, flattened_sample.device)
            dist_feat = self.storm_transformer(flattened_sample, actions2onehot(actions, self.action_dims), temporal_mask)
            
            # prior dit head
            _dist_feat = rearrange(dist_feat, "B L C -> (B L) C") 
            prior_logits = self._prior_dist_head(_dist_feat)
            prior_logits = rearrange(prior_logits, "(B L) K C -> B L K C",B=batch_size) 
            
            # decoding reward and termination with dist_feat
            reward_hat = self.reward_decoder(dist_feat)
            termination_hat = self.termination_decoder(dist_feat)

            # env loss
            reconstruction_loss = self.mse_loss_func(emb_hat, emb)
            reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
            termination_loss = self.bce_with_logits_loss_func(termination_hat, termination)
            # dyn-rep loss
            dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].detach(), prior_logits[:, :-1])
            representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:], prior_logits[:, :-1].detach())
            total_loss = reconstruction_loss + reward_loss + termination_loss + 0.5*dynamics_loss + 0.1*representation_loss

        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        if logger is not None:
            logger.log("WorldModel/reconstruction_loss", reconstruction_loss.item())
            logger.log("WorldModel/reward_loss", reward_loss.item())
            logger.log("WorldModel/termination_loss", termination_loss.item())
            logger.log("WorldModel/dynamics_loss", dynamics_loss.item())
            logger.log("WorldModel/dynamics_real_kl_div", dynamics_real_kl_div.item())
            logger.log("WorldModel/representation_loss", representation_loss.item())
            logger.log("WorldModel/representation_real_kl_div", representation_real_kl_div.item())
            logger.log("WorldModel/total_loss", total_loss.item())
            if log_video:
                emb_hat = symexp(emb_hat) if self.symlog else emb_hat
                
                logger.log("Recon/sample_video", torch.clamp(obs[::batch_size//16], 0, 1).cpu().float().detach().numpy())
                logger.log("Recon/rec_jepa_video", torch.clamp(tensor_unormalize(self.jepa_decoder.decode_video(emb[::batch_size//16])), 0, 1).cpu().float().detach().numpy())
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                    logger.log("Recon/rec_vae_video", torch.clamp(tensor_unormalize(self.jepa_decoder.decode_video(emb_hat[::batch_size//16])), 0, 1).cpu().float().detach().numpy())
    def reconstruction(self, obs):
        self.eval()
        with torch.no_grad():
            batch_size, batch_length = obs.shape[:2]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                vae_obs = rearrange(obs, "B L C H W -> (B L) C H W")

                # encoding
                emb = self.jepa_encoder(vae_obs) # [Batch&Length Channels sqrt(patch) sqrt(patch)]
                post_logits = self._vae.encode(emb)
                sample = self._vae.sample(post_logits, sample_mode="random_sample")
                flattened_sample = self._vae.flatten_sample(sample)

                # decoding image
                emb_hat = self._vae.decode(sample)
                emb_hat = symexp(emb_hat) if self.symlog else emb_hat

                emb = rearrange(emb, "(B L) C H W -> B L C H W",B=batch_size)
                emb_hat = rearrange(emb_hat, "(B L) C H W -> B L C H W",B=batch_size)
                obs_hat = tensor_unormalize(self.jepa_decoder.decode_video(emb_hat))
        return obs, obs_hat
