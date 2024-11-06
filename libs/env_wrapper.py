import gym
import gymnasium
import numpy as np
from collections import deque

# MineDojo Gymnasium Wrapper
class MineDojoGymnasium(gymnasium.Env):
    def __init__(self, minedojo_env:gym.Wrapper, seed, skip=4):
        super().__init__()
        self.minedojo_env = minedojo_env
        self.skip = skip
        self._seed = seed

        self.observation_space = minedojo_env.observation_space
        self.action_space = minedojo_env.action_space

        self.obs_buffer = deque(maxlen=2)
        self._elapsed_steps = 0

    def step(self, action, aciton_mask=None):
        all_obs = []
        self.obs_buffer.clear()
        total_reward = 0

        for _ in range(self.skip):
            if aciton_mask == None:
                exe_action = action 
            else:
                exe_action = action if self._valid_action(action, aciton_mask) else self.minedojo_env.action_space.no_op()

            obs, reward, done, info = self.minedojo_env.step(exe_action)
            self._elapsed_steps += 1
            elapsed_steps = self._elapsed_steps

            aciton_mask = obs["masks"] if not aciton_mask == None else None
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
                    'masks': aciton_mask, 
                    'elapsed_steps':elapsed_steps,
                    'all_obs':all_obs
                }

    def reset(self):
        obs = self.minedojo_env.reset()
        self._elapsed_steps = 1
        self.minedojo_env.seed(self._seed)
        return obs['rgb'],\
            {
                'masks': obs["masks"], 
                'elapsed_steps':1,
                'all_obs':[obs['rgb']]
            }

    def close(self):
        self.minedojo_env.close()

    def render(self, mode='human'):
        return self.minedojo_env.render(mode=mode)
    
    def _valid_action(self, action, mask):
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
    