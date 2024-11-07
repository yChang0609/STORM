import gymnasium
import argparse
from tensorboardX import SummaryWriter
import cv2
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import copy
import colorama
import random
import json
import shutil
import pickle
import os

from utils.utils import seed_np_torch, Logger, load_config
from utils.replay_buffer import ReplayBuffer
import archive.env_wrapper as env_wrapper
import agents.agents as agents
from sub_models.functions_losses import symexp
# from sub_models.world_models import WorldModel, MSELoss
from sub_models.jepa_world_models import JEPABaseWorldModel as WorldModel



def process_visualize(img):
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 640))
    return img


def build_single_env(env_name, image_size):
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    return env


def build_vec_env(env_name, image_size, num_envs):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size):
        return lambda: build_single_env(env_name, image_size)
    env_fns = []
    env_fns = [lambda_generator(env_name, image_size) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


def eval_episodes(num_episode, env_name, max_steps, num_envs, image_size,
                  world_model: WorldModel, agent: agents.ActorCriticAgent):
    world_model.eval()
    agent.eval()
    vec_env = train.build_single_env(env_name, image_size, seed=1)
    print("Current env: " + colorama.Fore.YELLOW + f"{env_name}" + colorama.Style.RESET_ALL)
    sum_reward = np.zeros(num_envs)
    current_obs, current_info = vec_env.reset()
    action_mask = current_info['masks']
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    save_frames = []
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    fps = 30
    frame_size = (224, 224) 

    final_rewards = []
    # for total_steps in tqdm(range(max_steps//num_envs)):
    while True:
        # save_frames += [cv2.cvtColor(current_obs.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)]*4
        save_frames.extend([cv2.cvtColor(obs.transpose(1, 2, 0), cv2.COLOR_RGB2BGR) for obs in current_info['all_obs']])
        # sample part >>>
        with torch.no_grad():
            if len(context_action) == 0:
                action = vec_env.action_space.sample()
            else:
                context_latent,_ = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                model_context_action = np.stack(list(context_action), axis=0)
                model_context_action = torch.Tensor(model_context_action.reshape(1, *model_context_action.shape)).cuda() #[np.newaxis, 0, 1]
                prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                action = agent.sample_as_env_action(
                    torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                    greedy=False
                )
                action = np.squeeze(action)
        context_obs.append(rearrange(torch.Tensor(current_obs.copy()).cuda(), "C H W -> 1 1 C H W")/255)
        context_action.append(action)

        obs, reward, done, truncated, info = vec_env.step(action, action_mask)
        # cv2.imshow("current_obs", process_visualize(obs[0]))
        # cv2.waitKey(10)

        # update current_obs, current_info and sum_reward
        sum_reward += reward
        current_obs = obs
        current_info = info
        action_mask = info['masks']

        truncated = np.array([truncated])
        done = np.array([done])
        done_flag = np.logical_or(done, truncated)
        if done_flag.any():
            for i in range(num_envs):
                if done_flag:
                    final_rewards.append(sum_reward[i])
                    
                    # insert done_frame
                    done_frame = np.ones((224, 224, 3), dtype=np.uint8) * 255
                    text = f"Ep{len(final_rewards)}:{sum_reward[i]}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    text_x = (done_frame.shape[1] - text_width) // 2  
                    text_y = (done_frame.shape[0] + text_height) // 2  
                    position = (text_x, text_y)  
                    cv2.putText(
                        done_frame, 
                        text, position, cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 0), 2,
                        cv2.LINE_AA      
                    )
                    save_frames += [done_frame]*16

                    # print(f"save_frames len - {len(final_rewards)}:{len(save_frames)} / {sum_reward[i]}")
                    sum_reward[i] = 0
                    current_obs, current_info = vec_env.reset()
                    action_mask = current_info['masks']
                    if len(final_rewards) == num_episode:
                        # save video
                        # save_frames += [cv2.cvtColor(current_obs.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)]*4
                        print("Mean reward: " + colorama.Fore.YELLOW + f"{np.mean(final_rewards)}" + colorama.Style.RESET_ALL)
                        out = cv2.VideoWriter(f"eval_result/MineDojo/episodes{num_episode}_{np.mean(final_rewards)}.mp4", fourcc, fps, frame_size)
                        for frame in save_frames:
                            out.write(frame)
                        out.release()
                        return np.mean(final_rewards)
        # <<< sample part


if __name__ == "__main__":
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-env_name", type=str, required=True)
    parser.add_argument("-run_name", type=str, required=True)
    args = parser.parse_args()
    conf = load_config(args.config_path)
    print(str(args))
    # print(colorama.Fore.RED + str(conf) + colorama.Style.RESET_ALL)

    # set seed
    seed_np_torch(seed=conf.BasicSettings.Seed)

    # build and load model/agent
    import train
    dummy_env = train.build_single_env(args.env_name, conf.BasicSettings.ImageSize,seed=1)
    action_dims = list(dummy_env.action_space.nvec)

    # build world model and agent
    world_model = train.build_world_model(conf, action_dims)
    agent = train.build_agent(conf, action_dims)
    root_path = f"ckpt/{args.run_name}"

    import glob
    pathes = glob.glob(f"{root_path}/world_model_*.pth")
    steps = [int(path.split("_")[-1].split(".")[0]) for path in pathes]
    steps.sort()
    steps = steps[-1:]
    print(steps)
    results = []
    for step in tqdm(steps):
        world_model.load_state_dict(torch.load(f"{root_path}/world_model_{step}.pth"))
        agent.load_state_dict(torch.load(f"{root_path}/agent_{step}.pth"))
        # # eval
        episode_avg_return = eval_episodes(
            num_episode=20,
            env_name=args.env_name,
            num_envs=1,
            max_steps=conf.JointTrainAgent.SampleMaxSteps,
            image_size=conf.BasicSettings.ImageSize,
            world_model=world_model,
            agent=agent
        )
        results.append([step, episode_avg_return])
    with open(f"eval_result/{args.run_name}.csv", "w") as fout:
        fout.write("step, episode_avg_return\n")
        for step, episode_avg_return in results:
            fout.write(f"{step},{episode_avg_return}\n")
