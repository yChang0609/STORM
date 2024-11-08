import gymnasium
import argparse
import numpy as np
from einops import rearrange
import torch
from collections import deque
from tqdm import tqdm
import colorama
import shutil
import os
import yaml
import pprint

from utils.utils import seed_np_torch, Logger
from libs.env_wrapper import MineDojoGymnasium

from world_models.world_model_base import WorldModelBase
import agents.agents as agents
from utils.replay_buffer import ReplayBuffer

# MineDojo
from libs.mine_env import build_env

def build_single_env(params, seed = None)->gymnasium.Wrapper:
    env = build_env(params, seed)
    env = MineDojoGymnasium(
        minedojo_env=env,
        skip=1,
        seed=seed
    )
    return env

def build_vec_env(env_name, image_size, num_envs, seed):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size):
        return lambda: build_single_env(env_name, image_size, seed)
    env_fns = []
    env_fns = [lambda_generator(env_name, image_size) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env

def train_world_model_step(replay_buffer: ReplayBuffer, world_model: WorldModelBase, batch_size, demonstration_batch_size, batch_length, logger):
    obs, action, reward, termination = replay_buffer.sample(batch_size, demonstration_batch_size, batch_length)
    world_model.update(obs, action, reward, termination, logger=logger)

@torch.no_grad()
def world_model_imagine_data(replay_buffer: ReplayBuffer,
                             world_model: WorldModelBase, agent: agents.ActorCriticAgent,
                             imagine_batch_size, imagine_demonstration_batch_size,
                             imagine_context_length, imagine_batch_length,
                             log_video, logger):
    '''
    Sample context from replay buffer, then imagine data with world model and agent
    '''
    world_model.eval()
    agent.eval()

    sample_obs, sample_action, sample_reward, sample_termination = replay_buffer.sample(
        imagine_batch_size, imagine_demonstration_batch_size, imagine_context_length)
    latent, action, reward_hat, termination_hat = world_model.imagine_data(
        agent, sample_obs, sample_action,
        imagine_batch_size=imagine_batch_size+imagine_demonstration_batch_size,
        imagine_batch_length=imagine_batch_length,
        log_video=log_video,
        logger=logger
    )
    return latent, action, None, None, reward_hat, termination_hat

def joint_train_world_model_agent(params, 
                                  replay_buffer: ReplayBuffer,
                                  world_model: WorldModelBase, 
                                  agent: agents.ActorCriticAgent,
                                  max_steps, num_envs, image_size,
                                  train_dynamics_every_steps, train_agent_every_steps,
                                  batch_size,  batch_length,
                                  demonstration_batch_size, imagine_demonstration_batch_size,
                                  imagine_batch_size, imagine_context_length, imagine_batch_length,
                                  save_every_steps, 
                                  seed, logger
                                ):
    # create ckpt dir
    os.makedirs(f"ckpt/{args.log}", exist_ok=True)

    # build vec env, not useful in the Atari100k setting
    # but when the max_steps is large, you can use parallel envs to speed up
    env_name = params["Environment"]["task"]
    print("Current env: " + colorama.Fore.YELLOW + f"{env_name}" + colorama.Style.RESET_ALL)
    vec_env = build_single_env(params)

    
    # reset envs and variables
    sum_reward = np.zeros(num_envs)
    current_obs, current_info = vec_env.reset()
    # action_mask = current_info['masks']

    # init context qeue
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    # sample and train
    for total_steps in tqdm(range(max_steps//num_envs)):
        # sample part >>>
        if replay_buffer.ready():
            world_model.eval()
            agent.eval()
            with torch.no_grad():
                if len(context_action) == 0:
                    action = vec_env.action_space.sample()
                else:
                    context_latent, _ = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                    model_context_action = np.stack(list(context_action), axis=0)
                    model_context_action = torch.Tensor(model_context_action.reshape(1, *model_context_action.shape)).cuda() #[np.newaxis, 0, 1]
                    prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                    action = agent.sample_as_env_action(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                        greedy=False
                    )
                    action = np.squeeze(action)
            # if(len(context_obs) == 16):
            #     logger.log("Imagine/test_video", torch.clamp(torch.cat(list(context_obs), dim=1), 0, 1).cpu().float().detach().numpy())

            context_obs.append(rearrange(torch.Tensor(current_obs.copy()).cuda(), "C H W -> 1 1 C H W")/255) # [one env , len obs ,(obs) ]
            context_action.append(action)

        else:
            action = vec_env.action_space.sample()

        obs, reward, done, truncated, info = vec_env.step(action)
        # print(current_obs.shape)
        # logger.log("Sample/test_images", current_obs[np.newaxis,:,:,:]/255)
        replay_buffer.append(current_obs, action, reward, done)

        # update current_obs, current_info and sum_reward
        sum_reward += reward
        current_obs = obs
        current_info = info
        # action_mask = info['masks']

        truncated = np.array([truncated])
        done = np.array([done])
        done_flag = np.logical_or(done, truncated)
        if done_flag.any() :
            for i in range(num_envs):
                if done_flag:
                    logger.log(f"sample/{env_name}_reward", sum_reward[i])
                    logger.log(f"sample/{env_name}_episode_steps", current_info["elapsed_steps"]//4)
                    logger.log("replay_buffer/length", len(replay_buffer))
                    sum_reward[i] = 0
                    current_obs, current_info = vec_env.reset()
                    # action_mask = current_info['masks']
        # <<< sample part
        
        # train world model part >>>
        if replay_buffer.ready() and total_steps % (train_dynamics_every_steps//num_envs) == 0:
            train_world_model_step(
                replay_buffer=replay_buffer,
                world_model=world_model,
                batch_size=batch_size,
                demonstration_batch_size=demonstration_batch_size,
                batch_length=batch_length,
                logger=logger
            )
        # <<< train world model part

        # train agent part >>>
        if replay_buffer.ready() and total_steps % (train_agent_every_steps//num_envs) == 0 and total_steps*num_envs >= 0:
            if total_steps % (save_every_steps//num_envs) == 0:
                log_video = True
            else:
                log_video = False

            imagine_latent, agent_action, agent_logprob, agent_value, imagine_reward, imagine_termination = world_model_imagine_data(
                replay_buffer=replay_buffer,
                world_model=world_model,
                agent=agent,
                imagine_batch_size=imagine_batch_size,
                imagine_demonstration_batch_size=imagine_demonstration_batch_size,
                imagine_context_length=imagine_context_length,
                imagine_batch_length=imagine_batch_length,
                log_video=log_video,
                logger=logger
            )

            agent.update(
                latent=imagine_latent,
                action=agent_action,
                old_logprob=agent_logprob,
                old_value=agent_value,
                reward=imagine_reward,
                termination=imagine_termination,
                logger=logger
            )
        # <<< train agent part

        # save model per episode
        if total_steps % (save_every_steps//num_envs) == 0:
            # print(colorama.Fore.GREEN + f"Saving model at total steps {total_steps}" + colorama.Style.RESET_ALL)
            print(f"Saving model at total steps {total_steps}")
            torch.save(world_model.state_dict(), f"ckpt/{args.log}/world_model_{total_steps}.pth")
            torch.save(agent.state_dict(), f"ckpt/{args.log}/agent_{total_steps}.pth")

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

def build_replay_buffer(params, action_dims):
    return ReplayBuffer(
        obs_shape=(params["BasicSettings"]["ImageSize"], params["BasicSettings"]["ImageSize"], 3),
        action_dim=action_dims,
        num_envs=params["JointTrainAgent"]["NumEnvs"],
        max_length=params["JointTrainAgent"]["BufferMaxLength"],
        warmup_length=params["JointTrainAgent"]["BufferWarmUp"],
        store_on_gpu=params["BasicSettings"]["ReplayBufferOnGPU"],
    )
if __name__ == "__main__":
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-log", type=str, required=True)
    parser.add_argument("-seed", type=int, required=True)
    parser.add_argument("-config", type=str, required=True)
    args = parser.parse_args()
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)

    # load config file to params
    # conf = load_config(args.config)
    params = None
    with open(args.config, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        print('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    # set seed
    seed_np_torch(seed=args.seed)

    # tensorboard writer
    logger = Logger(path=f"runs/{args.log}")

    # copy config file
    shutil.copy(args.config, f"runs/{args.log}/config.yaml")

    # distinguish between tasks, other debugging options are removed for simplicity
    training_task = params["Task"]
    if training_task == "JointTrainAgent":
        # getting action_dim with dummy env
        dummy_env = build_single_env(params, seed=1)
        action_dims = list(dummy_env.action_space.nvec)
        # build world model and agent
        world_model = build_world_model(params, action_dims)
        agent = build_agent(params, action_dims)

        # build replay buffer
        replay_buffer = build_replay_buffer(params, action_dims)
        
        # judge whether to load demonstration trajectory
        if params["JointTrainAgent"]["UseDemonstration"]:
            # print(colorama.Fore.MAGENTA + f"loading demonstration trajectory from {args.trajectory_path}" + colorama.Style.RESET_ALL)
            print(f"loading demonstration trajectory from {args.trajectory_path}")
            replay_buffer.load_trajectory(path=args.trajectory_path)
        # train
        joint_train_world_model_agent(
            params=params,
            replay_buffer=replay_buffer,
            world_model=world_model,
            agent=agent,
            num_envs=params["JointTrainAgent"]["NumEnvs"],
            max_steps=params["JointTrainAgent"]["SampleMaxSteps"],
            image_size=params["BasicSettings"]["ImageSize"],
            train_dynamics_every_steps=params["JointTrainAgent"]["TrainDynamicsEverySteps"],
            train_agent_every_steps=params["JointTrainAgent"]["TrainAgentEverySteps"],
            batch_size=params["JointTrainAgent"]["BatchSize"],
            demonstration_batch_size=params["JointTrainAgent"]["DemonstrationBatchSize"] if params["JointTrainAgent"]["UseDemonstration"] else 0,
            batch_length=params["JointTrainAgent"]["BatchLength"],
            imagine_batch_size=params["JointTrainAgent"]["ImagineBatchSize"],
            imagine_demonstration_batch_size=params["JointTrainAgent"]["ImagineDemonstrationBatchSize"] if params["JointTrainAgent"]["UseDemonstration"] else 0,
            imagine_context_length=params["JointTrainAgent"]["ImagineContextLength"],
            imagine_batch_length=params["JointTrainAgent"]["ImagineBatchLength"],
            save_every_steps=params["JointTrainAgent"]["SaveEverySteps"],
            seed=params["Environment"]["seed"],
            logger=logger
        )
    else:
        raise NotImplementedError(f"Task {training_task} not implemented")
