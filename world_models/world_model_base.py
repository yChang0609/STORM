import torch
import torch.nn as nn
from agents.agents import ActorCriticAgent

class WorldModelBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1

        self.latent_buffer = None
        self.hidden_buffer = None
        self.action_buffer = None
        self.reward_hat_buffer = None
        self.termination_hat_buffer = None

    def init_imagine_buffer(self, stoch_flattened_dim, transformer_hidden_dim, action_dims,
                            imagine_batch_size, imagine_batch_length, dtype):
        '''
            This can slightly improve the efficiency of imagine data But may vary across different machines
        '''
        if self.imagine_batch_size != imagine_batch_size or self.imagine_batch_length != imagine_batch_length:
            print(f"init_imagine_buffer: {imagine_batch_size}x{imagine_batch_length}@{dtype}")

            self.imagine_batch_size = imagine_batch_size
            self.imagine_batch_length = imagine_batch_length

            latent_size = (imagine_batch_size, imagine_batch_length+1, stoch_flattened_dim)
            hidden_size = (imagine_batch_size, imagine_batch_length+1, transformer_hidden_dim)
            action_size = (imagine_batch_size, imagine_batch_length, len(action_dims))
            scalar_size = (imagine_batch_size, imagine_batch_length)

            self.latent_buffer = torch.zeros(latent_size, dtype=dtype, device="cuda")
            self.hidden_buffer = torch.zeros(hidden_size, dtype=dtype, device="cuda")
            self.action_buffer = torch.zeros(action_size, dtype=dtype, device="cuda")
            self.reward_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")
            self.termination_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")

    def encode_obs(self, obs):
        raise NotImplementedError("Subclasses must implement the method.")
    
    def calc_last_dist_feat(self, latent, actions):
        raise NotImplementedError("Subclasses must implement the method.")
    
    def predict_next(self, last_flattened_sample, actions, log_video=True):
        raise NotImplementedError("Subclasses must implement the method.")

    def imagine_data(self, 
                     agent: ActorCriticAgent, 
                     sample_obs, sample_action,
                     imagine_batch_size, imagine_batch_length, 
                     log_video, logger):
        raise NotImplementedError("Subclasses must implement the method.")
        
    def update(self, obs, actions, reward, termination, logger=None):
        raise NotImplementedError("Subclasses must implement the method.")