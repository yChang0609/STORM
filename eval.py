import glob
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



mount_path_env = os.getenv('MOUNT_PATH', "")

def load_images(folder_path, start=0, end=-1):
    files = sorted(os.listdir(folder_path))
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if end == -1:
        end = len(image_files)
    
    selected_files = image_files[start:end]
    
    images = []
    for file_name in selected_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            with Image.open(file_path) as img:
                image = cv2.resize(np.array(img), (224, 224), interpolation=cv2.INTER_LINEAR)
                image_tensor = torch.Tensor(image).permute(2, 0, 1).cuda()  # (H, W, C) -> (C, H, W)
                image_tensor = rearrange(image_tensor, "C H W -> 1 1 C H W") / 255.0
                images.append(image_tensor)
        except Exception as e:
            print(f"failed load {file_name}: {e}")
    
    return images


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

def eval_agent(
        num_episode, 
        params, num_envs, 
        world_model: WorldModelBase, agent: agents.ActorCriticAgent, 
        seed=456, verbose=False):
    name = params["Environment"]["task"]
    if verbose: print("Current env: " + colorama.Fore.YELLOW + f"{name}" + colorama.Style.RESET_ALL)
    world_model.eval()
    agent.eval()

    vec_env = build_single_env(params, seed=seed)
    current_obs, current_info = vec_env.reset()

    episode_step = 0
    sum_reward = 0
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    success_count = 0
    final_ep_steps = []
    final_rewards = []
    if verbose:
        fps = 30
        save_frames = []
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        frame_size = tuple(reversed(params["Environment"]["task_parameter"]["image_size"]))

    while True:
        if verbose: save_frames.extend([cv2.cvtColor(obs.transpose(1, 2, 0), cv2.COLOR_RGB2BGR) for obs in current_info['all_obs']])
        with torch.no_grad():
            if len(context_action) == 0:
                action = vec_env.action_space.sample()
            else:
                context_latent,_ = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                model_context_action = np.stack(list(context_action), axis=0)
                model_context_action = torch.Tensor(model_context_action.reshape(1, *model_context_action.shape)).cuda() #[np.newaxis, 0, 1]
                prior_flattened_sample, last_dist_feat, jepa_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                action = agent.sample_as_env_action(
                    torch.cat([prior_flattened_sample, last_dist_feat] if jepa_feat==None else [prior_flattened_sample, last_dist_feat, jepa_feat], dim=-1),
                    greedy=False
                )
                action = np.squeeze(action)
        context_obs.append(rearrange(torch.Tensor(current_obs.copy()).cuda(), "C H W -> 1 1 C H W")/255)
        context_action.append(action)
        obs, reward, done, truncated, info = vec_env.step(action)

        episode_step += 1
        sum_reward += reward
        current_obs = obs
        current_info = info
        if done:    
            final_ep_steps.append(episode_step)
            final_rewards.append(sum_reward)
            if verbose: 
                print(f"Episode-{len(final_rewards)} / step: {episode_step} & reward: {sum_reward}")
                # insert done_frame
                done_frame = np.ones((frame_size[0], frame_size[1], 3), dtype=np.uint8) * 255
                text = f"Ep{len(final_rewards)}:{sum_reward:.2f}"
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

            if episode_step < params["Environment"]["task_parameter"]["max_episode_len"]: #sum_reward > 10:
                success_count += 1
            episode_step = 0
            sum_reward = 0

            if len(final_rewards) == num_episode:
                if verbose: 
                    # save video
                    print(f"success_rate: {(success_count / num_episode * 100):.2f}%")
                    print("Mean reward: " + f"{np.mean(final_rewards):.4f}")
                    os.makedirs(f"{eval_result_path}/videos", exist_ok=True)
                    out = cv2.VideoWriter(f"{eval_result_path}/videos/{seed}-episodes{num_episode}_{np.mean(final_rewards):.2f}.mp4", fourcc, fps, frame_size)
                    for frame in save_frames:
                        out.write(frame)
                    out.release()
                vec_env.close()
                return final_ep_steps, final_rewards, (success_count / num_episode * 100)
            
            current_obs, current_info = vec_env.reset()


        
def collect_data(params, world_model: WorldModelBase, agent: agents.ActorCriticAgent,  seed=456):
    name = params["Environment"]["task"]
    print("Current env: " + colorama.Fore.YELLOW + f"{name}" + colorama.Style.RESET_ALL)

    world_model.eval()
    agent.eval()

    vec_env = build_single_env(params, seed=seed)
    current_obs, current_info = vec_env.reset()
    
    living = True
    replay_data = []
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
                prior_flattened_sample, last_dist_feat, jepa_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                action = agent.sample_as_env_action(
                    torch.cat([prior_flattened_sample, last_dist_feat] if jepa_feat==None else [prior_flattened_sample, last_dist_feat, jepa_feat], dim=-1),
                    greedy=False
                )
                action = np.squeeze(action)
        context_obs.append(rearrange(torch.Tensor(current_obs.copy()).cuda(), "C H W -> 1 1 C H W")/255)
        replay_data.append(rearrange(torch.Tensor(current_obs.copy()).cuda(), "C H W -> 1 1 C H W")/255)
        context_action.append(action)
        obs, reward, done, truncated, info = vec_env.step(action)
        current_obs = obs
        current_info = info
        living = (not done) or (current_info['elapsed_steps'] > 125)
    vec_env.close()
    return replay_data
    

def eval_reconstruction(step, replay_data, world_model: WorldModelBase, export_path, sequence=False):
    world_model.eval()
    # agent.eval()
    recon_list = replay_data if sequence else replay_data[-1]
    obs, obs_hat = world_model.reconstruction(torch.cat(recon_list, dim=1))
    # save_tensor_with_channels(obs, f"{export_path}/{step}_reconstruction/", f"original")
    save_tensor_with_channels(obs_hat, f"{export_path}/{step}_reconstruction/", f"vae-recon")


def load_model(log_file, only_wm=False):
    ckpt_path = os.path.join(mount_path_env,"ckpt/")
    config = os.path.join("runs", log_file, "config.yaml")
    params = load_config(config)
    # build and load model/agent
    # import train
    dummy_env = build_single_env(params, seed=1)
    action_dims = list(dummy_env.action_space.nvec)
    dummy_env.close()
    # build world model and agent
    world_model = build_world_model(params, action_dims)
    agent = None if only_wm else build_agent(params, action_dims) 

    root_path = os.path.join(ckpt_path,log_file)
    pathes = glob.glob(f"{root_path}/world_model_*.pth")
    steps = [int(path.split("_")[-1].split(".")[0]) for path in pathes]
    steps.sort()
    load_step = steps[-1]
    
    world_model.load_state_dict(torch.load(f"{root_path}/world_model_{load_step}.pth"))
    if not only_wm: agent.load_state_dict(torch.load(f"{root_path}/agent_{load_step}.pth"))

    return world_model, agent, params

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-logs", 
        nargs='+', 
        help="Log file names",
        required=True
    )
    parser.add_argument("-mode", type=str, required=False,default="agent")
    parser.add_argument("-reply_data", type=str, required=False)

    args = parser.parse_args()

    print(str(args))

    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    num_episode = 20
    assert len(args.logs) > 0
    if "reconstruction" in args.mode :
        if args.reply_data == None:
            dummy_world_model, dummy_agent, dummy_params = load_model(args.logs[0])
            replay_data = collect_data(
                params=dummy_params,
                world_model=dummy_world_model,
                agent=dummy_agent,
                seed=123
            )
            del dummy_world_model, dummy_agent, dummy_params
        else:
            replay_data = load_images(args.reply_data)

    for log in args.logs:
        eval_result_path = os.path.join(mount_path_env,f"eval_result/{log}")
        os.makedirs(eval_result_path, exist_ok=True)
        
        if args.mode == "agent":
            world_model, agent, params = load_model(log)
            episode_returns = []
            results = []
            success_rate_list = []
            seed_np_torch(seed=params["BasicSettings"]["Seed"])
            eval_seed_list = [456, 789, 357, 468, 790]
            for seed in eval_seed_list:
                episode_steps, episode_rewards, success_rate = eval_agent(
                    num_episode=num_episode,
                    params=params,
                    num_envs=1,
                    world_model=world_model,
                    agent=agent,
                    seed=seed
                )
                episode_returns.append([[i, episode_rewards[i], episode_steps[i]] for i in range(len(episode_rewards))])
                results.append([np.mean(episode_rewards), success_rate])
            write_file = os.path.join(eval_result_path, "return.csv")
            success_rate_temp = 0
            with open(write_file, "w+") as fout:
                for i in range(len(results)):
                    fout.write("episode, episode_return, episode_steps\n")
                    for ep, reward, ep_step in episode_returns[i]:
                        fout.write(f"{ep},{reward},{ep_step}\n")
                    fout.write("step, episode_avg_return, success_rate\n")
                    episode_avg_return, success_rate = results[i]
                    fout.write(f"{episode_avg_return}, {success_rate}\n")
                    success_rate_temp += success_rate 
                    fout.write("-----------------------------------\n")
                fout.write(f"Eval success rate:{success_rate_temp/len(eval_seed_list)}\n")
            print(f"Eval avg success rate: {success_rate_temp/len(eval_seed_list):.2f}%")
        elif "reconstruction" in args.mode:
            world_model, _, params = load_model(log,only_wm=True)
            eval_reconstruction(
                    replay_data=replay_data,
                    world_model=world_model,
                    export_path=eval_result_path,
                    sequence="clip" in args.mode
            )
