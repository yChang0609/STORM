import gym
import gymnasium
import numpy as np
from collections import deque

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

# MineDojo Gymnasium Wrapper
class MineDojoGymnasium(gymnasium.Env):
    def __init__(self, minedojo_env:gym.Wrapper, seed, skip=4):
        super().__init__()
        self.minedojo_env = minedojo_env
        self.skip = skip
        self._seed = seed

        self.observation_space = minedojo_env.observation_space["rgb"]
        self.action_space = minedojo_env.action_space

        self.obs_buffer = deque(maxlen=2)
        self._elapsed_steps = 0

    def step(self, action):
        all_obs = []
        self.obs_buffer.clear()
        total_reward = 0

        for _ in range(self.skip):
            obs, reward, done, info = self.minedojo_env.step(action)
            self._elapsed_steps += 1
            elapsed_steps = self._elapsed_steps
            self.obs_buffer.append(obs['rgb'])
            all_obs.append(obs['rgb'])
            
            total_reward += reward
            if done:
                self._elapsed_steps = 0
                break
        obs_image = self.obs_buffer[0] if len(self.obs_buffer) == 1 else np.max(np.stack(self.obs_buffer), axis=0)
        
        truncated = False
        return obs_image, total_reward, done, truncated, \
                {
                    'elapsed_steps':elapsed_steps,
                    'all_obs':all_obs
                }

    def reset(self, **kwargs):
        obs = self.minedojo_env.reset()
        self._elapsed_steps = 1
        self.minedojo_env.seed(self._seed)
        return obs['rgb'],\
            {
                'elapsed_steps':1,
                'all_obs':[obs['rgb']]
            }

    def close(self):
        self.minedojo_env.close()

    def render(self, mode='human'):
        return self.minedojo_env.render(mode=mode)
    