import gymnasium
import argparse

import os

import cv2
import numpy as np
from einops import rearrange
import torch

from collections import deque
from tqdm import tqdm
import colorama
import yaml
import pprint

from utils.utils import seed_np_torch, load_config
from utils.build_model import *

from world_models.world_model_base import WorldModelBase
from agents import agents
from utils.build_model import build_agent, build_world_model
from libs.env_wrapper import build_single_env

from PIL import Image

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-log", type=str, required=True)
parser.add_argument("-seed", type=int, required=True)
parser.add_argument("-config", type=str, required=True)
parser.add_argument("-mode", type=str, required=False,default="agent")

args = parser.parse_args()

print(str(args))

mount_path_env = os.getenv('MOUNT_PATH', "")
ckpt_path = os.path.join(mount_path_env,"ckpt/")
eval_result_path = os.path.join(mount_path_env,f"eval_result/{args.log}")
os.makedirs(eval_result_path, exist_ok=True)


def save_tensor_with_channels(tensor, output_dir, base_file_name="original"):
    """
    Save a tensor with dynamic channels as images.

    Args:
        tensor (torch.Tensor): Input tensor of shape [B, C, H, W] or [B, N, C, H, W].
        output_dir (str): Directory to save the images.
        base_file_name (str): Base name for saved images.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Handle dimensions
    if len(tensor.shape) == 5:  # [B, N, C, H, W]
        batch_size, batch_len, _, height, width = tensor.shape
    elif len(tensor.shape) == 4:  # [B, C, H, W]
        tensor = tensor.unsqueeze(1)  # Add a dummy channel dimension
        batch_size, batch_len, _, height, width = tensor.shape
    else:
        raise ValueError("Unsupported tensor shape. Expected 4D or 5D tensor.")

    # Iterate over batch and channel dimensions
    assert batch_size == 1
    for batch_idx in range(batch_size):
        for idx in range(batch_len):
            # Extract individual image tensor [C, H, W]
            image_tensor = tensor[batch_idx, idx]
            
            # Permute to (H, W, C)
            image_tensor = image_tensor.permute(1, 2, 0)  # [H, W, C]

            # Normalize to range [0, 255]
            image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min()) * 255
            image_array = image_tensor.cpu().byte().numpy()

            # Convert to PIL image and save
            image = Image.fromarray(image_array)
            # file_name = f"step{idx}_{base_file_name}_batch{batch_idx}.png"
            file_name = f"step{idx}_{base_file_name}.png"
            image.save(os.path.join(output_dir, file_name))



def process_visualize(img):
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 640))
    return img


def eval_episodes(step, num_episode, params, num_envs, world_model: WorldModelBase, agent: agents.ActorCriticAgent, seed=456):
    name = params["Environment"]["task"]
    print("Current env: " + colorama.Fore.YELLOW + f"{name}" + colorama.Style.RESET_ALL)
    world_model.eval()
    agent.eval()

    vec_env = build_single_env(params, seed=seed)
    current_obs, current_info = vec_env.reset()
    
    sum_reward = np.zeros(num_envs)
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    fps = 30
    save_frames = []
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    frame_size = tuple(reversed(params["Environment"]["task_parameter"]["image_size"]))

    success_count = 0
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

        obs, reward, done, truncated, info = vec_env.step(action)
        # cv2.imshow("current_obs", process_visualize(obs[0]))
        # cv2.waitKey(10)

        # update current_obs, current_info and sum_reward
        sum_reward += reward
        current_obs = obs
        current_info = info

        truncated = np.array([truncated])
        done = np.array([done])
        done_flag = np.logical_or(done, truncated)
        if done_flag.any():
            for i in range(num_envs):
                if done_flag:
                    final_rewards.append(sum_reward[i])
                    
                    # insert done_frame
                    done_frame = np.ones((frame_size[0], frame_size[1], 3), dtype=np.uint8) * 255
                    text = f"Ep{len(final_rewards)}:{sum_reward[i]:.2f}"
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

                    if sum_reward[i] > 10:
                        success_count += 1
                    sum_reward[i] = 0
                    current_obs, current_info = vec_env.reset()
                    if len(final_rewards) == num_episode:
                        # save video
                        # save_frames += [cv2.cvtColor(current_obs.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)]*4
                        print(f"success_rate: {(success_count / num_episode * 100):.2f}%")
                        print("Mean reward: " + colorama.Fore.YELLOW + f"{np.mean(final_rewards)}" + colorama.Style.RESET_ALL)
                        os.makedirs(f"{eval_result_path}/{step}_videos", exist_ok=True)
                        out = cv2.VideoWriter(f"{eval_result_path}/{step}_videos/{seed}-episodes{num_episode}_{np.mean(final_rewards):.2f}.mp4", fourcc, fps, frame_size)
                        for frame in save_frames:
                            out.write(frame)
                        out.release()
                        vec_env.close()
                        return final_rewards, np.mean(final_rewards)
        # <<< sample part
        


def eval_reconstruction(step, params, world_model: WorldModelBase, agent: agents.ActorCriticAgent, seed=456, sequence=False):
    name = params["Environment"]["task"]
    print("Current env: " + colorama.Fore.YELLOW + f"{name}" + colorama.Style.RESET_ALL)

    world_model.eval()
    agent.eval()

    vec_env = build_single_env(params, seed=seed)
    current_obs, current_info = vec_env.reset()
    
    living = True
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    # sample
    while living:
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
        obs, reward, done, truncated, info = vec_env.step(action)
        current_obs = obs
        current_info = info
        living = (not done) or (current_info['elapsed_steps'] > 125)
    recon_list = list(context_obs) if sequence else list(context_obs[-1])
    obs, obs_hat = world_model.reconstruction(torch.cat(recon_list, dim=1))
    save_tensor_with_channels(obs, f"{eval_result_path}/{step}_reconstruction/", f"original")
    save_tensor_with_channels(obs_hat, f"{eval_result_path}/{step}_reconstruction/", f"vae-recon")

    vec_env.close()


if __name__ == "__main__":
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    params = None
    with open(args.config, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        print('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
    
    # set seed
    seed_np_torch(seed=params["BasicSettings"]["Seed"])

    # build and load model/agent
    # import train
    dummy_env = build_single_env(params, seed=1)
    action_dims = list(dummy_env.action_space.nvec)
    dummy_env.close()
    # build world model and agent
    world_model = build_world_model(params, action_dims)
    agent = build_agent(params, action_dims)
    root_path = f"{ckpt_path}/{args.log}"

    import glob
    pathes = glob.glob(f"{root_path}/world_model_*.pth")
    steps = [int(path.split("_")[-1].split(".")[0]) for path in pathes]
    steps.sort()
    steps = steps[-1:]
    print(steps)
    episode_returns = []
    results = []
    if args.mode == "agent":
        for step in tqdm(steps):
            world_model.load_state_dict(torch.load(f"{root_path}/world_model_{step}.pth"))
            agent.load_state_dict(torch.load(f"{root_path}/agent_{step}.pth"))
            # # eval
            eval_seed_list = [456, 789, 357, 468, 790]
            for seed in eval_seed_list:
                episode_rewards, episode_avg_return = eval_episodes(
                    step=step,
                    num_episode=20,
                    params=params,
                    num_envs=1,
                    world_model=world_model,
                    agent=agent,
                    seed=seed
                )
                episode_returns.append([[i, episode_rewards[i]] for i in range(len(episode_rewards))])
                results.append([step, episode_avg_return])
        write_file = os.path.join(eval_result_path, "return.csv")
        with open(write_file, "w+") as fout:
            for i in range(len(results)):
                fout.write("episode, episode_return\n")
                for ep, reward in episode_returns[i]:
                    fout.write(f"{ep},{reward}\n")
                fout.write("step, episode_avg_return\n")
                step, episode_avg_return = results[i]
                fout.write(f"{step},{episode_avg_return}\n")
                fout.write("-----------------------------------\n")
    elif "reconstruction" in args.mode:
        for step in tqdm(steps):
            world_model.load_state_dict(torch.load(f"{root_path}/world_model_{step}.pth"))
            agent.load_state_dict(torch.load(f"{root_path}/agent_{step}.pth"))
            eval_reconstruction(
                    step=step,
                    params=params,
                    world_model=world_model,
                    agent=agent,
                    seed=123,
                    sequence="clip" in args.mode
            )