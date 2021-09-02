import numpy as np
import torch
import os
import msgpack

class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(s, # self
                 capacity: int,  # total number of sample
                 obs_shape: tuple,  # observation shape
                 action_shape: tuple,  # action shape
                 obs_dtype: type = np.float32,  # observation dtype
                 action_dtype: type = np.float32,  # action dtype
                 device="cpu"  # cpu, cuda, cuda:0, etc.
                 ):
        s.capacity = capacity = int(capacity)
        s.device = device

        s.idx = 0
        s.full = False
        s.num_recent = -1  # int, most recent samples for updating
        
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        #obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        s.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        s.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        s.actions = np.empty((capacity, *action_shape), dtype=action_dtype)
        s.rewards = np.empty((capacity, 1), dtype=np.float32)
        s.not_dones = np.empty((capacity, 1), dtype=np.float32)
        s.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

    def __len__(s):
        return s.capacity if s.full else s.idx

    def add(s, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(s.obses[s.idx], obs)
        np.copyto(s.actions[s.idx], action)
        np.copyto(s.rewards[s.idx], reward)
        np.copyto(s.next_obses[s.idx], next_obs)
        np.copyto(s.not_dones[s.idx], not done)
        np.copyto(s.not_dones_no_max[s.idx], not done_no_max)

        s.idx = (s.idx + 1) % s.capacity
        s.full = s.full or s.idx == 0

    def sample(s, batch_size):
        """
        sample batch_size samples from the most recent num_recent samples
        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max
        """
        idx_end = s.idx
        if s.num_recent > batch_size:
            idx_begin = s.idx - s.num_recent
            s.num_recent = -1  # reset num_recent
        else:
            idx_begin = s.idx - s.capacity
        if ~s.full:
            idx_begin = max(idx_begin, 0)
        idxs = np.random.randint(idx_begin, idx_end, size=batch_size)

        obses = torch.as_tensor(s.obses[idxs], device=s.device).float()
        actions = torch.as_tensor(s.actions[idxs], device=s.device)
        rewards = torch.as_tensor(s.rewards[idxs], device=s.device)
        next_obses = torch.as_tensor(s.next_obses[idxs], device=s.device).float()
        not_dones = torch.as_tensor(s.not_dones[idxs], device=s.device)
        not_dones_no_max = torch.as_tensor(s.not_dones_no_max[idxs], device=s.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max

    def __sizeof__(s):
        """to estimate the size in memory in bytes"""
        return super().__sizeof__()+s.obses.__sizeof__() +\
            s.next_obses.__sizeof__()+s.actions.__sizeof__() +\
            s.rewards.__sizeof__()+s.not_dones.__sizeof__() + \
            s.not_dones_no_max.__sizeof__()
        
    @staticmethod
    def _saveArr(path,arr):
        """ helper function to save numpy arr to path"""
        with open(path, 'wb+') as fh:
            k = msgpack.pack((f"{arr.dtype}",arr.shape),fh) # msgpack pack dtype and shape
            fh.write(arr.data) # write raw buffer to file
    
    @staticmethod
    def _loadArr(path):
        """ helper function to load numpy arr from path"""
        with open(path, 'rb') as fh:
            unpacker = msgpack.Unpacker(fh,use_list=False)
            dtype, shape = unpacker.unpack() # unpack dtype and shape
            fh.seek(unpacker.tell()) # go to raw buffer position
            return np.fromfile(fh, dtype=dtype).reshape(shape)

    def save(s,dir_path):
        """ save replay buffer to directory path dir_path"""
        dir_path = os.path.abspath(dir_path) # convert to absolute path
        try:os.mkdir(dir_path)
        except OSError:pass
        print(f"saving replay buffer to {dir_path}")
        s._saveArr(os.path.join(dir_path,"obses.npy"),s.obses)
        s._saveArr(os.path.join(dir_path,"next_obses.npy"),s.next_obses)
        s._saveArr(os.path.join(dir_path,"actions.npy"),s.actions)
        s._saveArr(os.path.join(dir_path,"rewards.npy"),s.rewards)
        s._saveArr(os.path.join(dir_path,"not_dones.npy"),s.not_dones)
        s._saveArr(os.path.join(dir_path,"not_dones_no_max.npy"),s.not_dones_no_max)
        with open(os.path.join(dir_path,"extra.msgpack"), 'wb') as file:
            msgpack.pack((s.capacity,s.idx,s.full), file)

    def load(s,dir_path):
        """load replay buffer from directory path dir_path"""
        dir_path = os.path.abspath(dir_path) # convert to absolute path
        s.obses=s._loadArr(os.path.join(dir_path,"obses.npy"))
        s.next_obses=s._loadArr(os.path.join(dir_path,"next_obses.npy"))
        s.actions=s._loadArr(os.path.join(dir_path,"actions.npy"))
        s.rewards=s._loadArr(os.path.join(dir_path,"rewards.npy"))
        s.not_dones=s._loadArr(os.path.join(dir_path,"not_dones.npy"))
        s.not_dones_no_max=s._loadArr(os.path.join(dir_path,"not_dones_no_max.npy"))
        with open(os.path.join(dir_path,"extra.msgpack"), 'rb') as file:
            s.capacity,s.idx,s.full = msgpack.unpack(file)

#     def save(s,dir_path):
#         """ save replay buffer, file should use extension .npz"""
#         try:os.mkdir(dir_path)
#         except OSError:pass
#         np.save(os.path.join(dir_path,"obses.npy"),s.obses, allow_pickle=False, fix_imports=False)
#         np.save(os.path.join(dir_path,"next_obses.npy"),s.next_obses, allow_pickle=False, fix_imports=False)
#         np.save(os.path.join(dir_path,"actions.npy"),s.actions, allow_pickle=False, fix_imports=False)
#         np.save(os.path.join(dir_path,"rewards.npy"),s.rewards, allow_pickle=False, fix_imports=False)
#         np.save(os.path.join(dir_path,"not_dones.npy"),s.not_dones, allow_pickle=False, fix_imports=False)
#         np.save(os.path.join(dir_path,"not_dones_no_max.npy"),s.not_dones_no_max)
#         with open(os.path.join(dir_path,"extra.msgpack"), 'wb') as file:
#             msgpack.pack((s.capacity,s.device,s.idx,s.full), file)