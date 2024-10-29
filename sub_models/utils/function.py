import torch
import torch.nn as nn

def action_onehot_function(actions, action_dims):
    ret = []
    for action, dim in zip(actions, action_dims):
        ret.append(nn.functional.one_hot(action.long(), num_classes=dim))
    return ret

def actions2onehot(actions ,action_dims):
    bl_vec = []
    for l_actions in actions: # [B ,L, action]
        l_vec = []
        for action in l_actions: # [L, action]
            l_vec.append(torch.cat(action_onehot_function(action,action_dims), dim=0))
        bl_vec.append(torch.stack(l_vec, dim=0))
    return torch.stack(bl_vec, dim=0)