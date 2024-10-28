import gym
import gymnasium
import numpy as np
from collections import deque
import cv2
from einops import rearrange
import copy

# MineDojo Gymnasium Wrapper
class MineDojoGymnasium(gymnasium.Env):
    def __init__(self, minedojo_env:gym.Wrapper, seed, skip=4):
        super().__init__()
        self.minedojo_env = minedojo_env
        self.skip = skip
        self.seed = seed
        self.observation_space = minedojo_env.observation_space
        self.action_space = minedojo_env.action_space

    def step(self, action, aciton_mask):
        total_reward = 0
        self.obs_buffer = deque(maxlen=2)
        all_obs = []
        for _ in range(self.skip):
            exe_action = action if self.valid_action(action, aciton_mask) else self.minedojo_env.action_space.no_op()
            obs, reward, done, info = self.minedojo_env.step(exe_action)
            aciton_mask = obs["masks"]
            self.obs_buffer.append(obs['rgb'])
            all_obs.append(obs['rgb'])
            total_reward += reward
            if done:
                break
        if len(self.obs_buffer) == 1:
            obs_image = self.obs_buffer[0]
        else:
            obs_image = np.max(np.stack(self.obs_buffer), axis=0)
        truncated = False
        return obs_image, total_reward, done, truncated, \
                {
                    'masks': aciton_mask, 
                    'elapsed_steps':info['elapsed_steps'],
                    'all_obs':all_obs
                }

    def reset(self):
        self.minedojo_env.seed(self.seed)
        obs = self.minedojo_env.reset()
        return obs['rgb'],\
            {
                'masks': obs["masks"], 
                'elapsed_steps':1,
                'all_obs':[obs['rgb']]
            }

    def render(self, mode='human'):
        return self.minedojo_env.render(mode=mode)

    def close(self):
        self.minedojo_env.close()

    def valid_action(self, action, mask):
        ret = False
        if (action[5] > 3):
            if(mask["action_type"][action[5]]):
                if(action[5] == 4 ): # functional actions 'craft'
                    if(mask["craft_smelt"][action[6]]):
                        ret = True
                elif(action[5] == 5): # functional actions 'equip'
                    if(mask["equip"][action[7]]):
                        ret = True
                elif(action[5] == 6): # functional actions 'place'
                    if(mask["place"][action[7]]):
                        ret = True
                elif(action[5] == 7): # functional actions 'destroy'
                    if(mask["destroy"][action[7]]):
                        ret = True
        else:
            ret = True

        return ret
    
    def limit_action(self, action):
        limit_values = [10, 11, 12, 13, 14]
        action[3] = limit_values[action[3] // len(limit_values)]
        return action


class LifeLossInfo(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives_info = None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        current_lives_info = info["lives"]
        if current_lives_info < self.lives_info:
            info["life_loss"] = True
            self.lives_info = info["lives"]
        else:
            info["life_loss"] = False

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.lives_info = info["lives"]
        info["life_loss"] = False
        return observation, info


class SeedEnvWrapper(gymnasium.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        self.seed = seed
        self.env.action_space.seed(seed)

    def reset(self, **kwargs):
        kwargs["seed"] = self.seed
        obs, _ = self.env.reset(**kwargs)
        return obs, _

    def step(self, action):
        return self.env.step(action)


class MaxLast2FrameSkipWrapper(gymnasium.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs, _

    def step(self, action):
        total_reward = 0
        self.obs_buffer = deque(maxlen=2)
        for _ in range(self.skip):
            obs, reward, done, truncated, info = self.env.step(action)
            self.obs_buffer.append(obs)
            total_reward += reward
            if done or truncated:
                break
        if len(self.obs_buffer) == 1:
            obs = self.obs_buffer[0]
        else:
            obs = np.max(np.stack(self.obs_buffer), axis=0)
        return obs, total_reward, done, truncated, info

def build_single_env(env_name, image_size):
    env = gymnasium.make(env_name, full_action_space=True, frameskip=1)
    from gymnasium.wrappers import AtariPreprocessing
    env = AtariPreprocessing(env, screen_size=image_size, grayscale_obs=False)
    return env


def build_vec_env(env_list, image_size, num_envs):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    assert num_envs % len(env_list) == 0
    env_fns = []
    vec_env_names = []
    for env_name in env_list:
        def lambda_generator(env_name, image_size):
            return lambda: build_single_env(env_name, image_size)
        env_fns += [lambda_generator(env_name, image_size) for i in range(num_envs//len(env_list))]
        vec_env_names += [env_name for i in range(num_envs//len(env_list))]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env, vec_env_names


if __name__ == "__main__":
    vec_env, vec_env_names = build_vec_env(['ALE/Pong-v5', 'ALE/IceHockey-v5', 'ALE/Breakout-v5', 'ALE/Tennis-v5'], 64, num_envs=8)
    current_obs, _ = vec_env.reset()
    while True:
        action = vec_env.action_space.sample()
        obs, reward, done, truncated, info = vec_env.step(action)
        # done = done or truncated
        if done.any():
            print("---------")
            print(reward)
            print(info["episode_frame_number"])
        cv2.imshow("Pong", current_obs[0])
        cv2.imshow("IceHockey", current_obs[2])
        cv2.imshow("Breakout", current_obs[4])
        cv2.imshow("Tennis", current_obs[6])
        cv2.waitKey(40)
        current_obs = obs
