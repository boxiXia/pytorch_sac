import glob
import numpy as np
import torch
import os
import msgpack



def _saveArr(path,arr):
    """ helper function to save a single numpy arr to path"""
    with open(path, 'wb+') as fh:
        k = msgpack.pack((f"{arr.dtype}",arr.shape),fh) # msgpack pack dtype and shape
        fh.write(arr.data) # write raw buffer to file

def _loadArr(path):
    """ helper function to load a single numpy arr from path"""
    with open(path, 'rb') as fh:
        unpacker = msgpack.Unpacker(fh,use_list=False)
        dtype, shape = unpacker.unpack() # unpack dtype and shape
        fh.seek(unpacker.tell()) # go to raw buffer position
        return np.fromfile(fh, dtype=dtype).reshape(shape)


class RecordArray:
    """wrapper class for fast save/load numpy array"""
    def __init__(s, # self
                 capacity: int,  # total number of sample
                 shape: tuple,  # shape of one sample
                 dtype: np.dtype = np.float32,
                 chunk_bytes:int = 1e6 # chunck sizes in bytes
                ):
        """initialize the class"""
        s._init_arr(capacity,shape,dtype,chunk_bytes)
        
        
    def _init_arr(s,capacity,shape,dtype,chunk_bytes):
        """helper to initialized the array"""
        s.capacity = int(capacity)
        try:len(shape)
        except: shape = (shape,)# zero-length
        s.shape = shape # sample shape
        s.arr = np.zeros((s.capacity, *shape), dtype=dtype) # actual data initialized with zero
        sample_rlen = int(np.prod(shape)) # sample raw length
        arr_rlen = s.capacity*sample_rlen # arr raw length
        # num of sample per chunk
        s.chunk_capacity = max(1,int(chunk_bytes/(s.arr.itemsize*sample_rlen)))
        s._chunk_bytes = s.chunk_capacity*s.arr.itemsize*sample_rlen        
        # total number of chunks
        num_chunk = int(np.ceil(s.capacity/s.chunk_capacity))
        
        # [0,s.chunk_capacity,2*s.chunk_capacity,3*s.chunk_capacity,.. capacity]
        s.chunk_bins = np.arange(num_chunk+1)*s.chunk_capacity
        s.chunk_bins[-1] = s.capacity
#         print(s.chunk_bins)
        
        assert(num_chunk*s.chunk_capacity>=s.capacity)
        # flag indicating whether chunk is modified
        s.chunk_modified = np.zeros(num_chunk,dtype=bool)
        s.chunk_saved = np.zeros(num_chunk,dtype=bool)
    
    @property
    def chunk_bytes(s):
        """return number of bytes per chunk"""
        return s._chunk_bytes
    
    def __len__(s):
        return s.capacity
    
    def __getitem__(s, idx):
        """get arr by idx"""
        return s.arr[idx]
    
    def __setitem__(s, idx, value):
        """set arr e.g. s[idx] = value"""
# #         assert(np.all(idx<s.capacity))
#         idx_bins = idx//s.chunk_capacity
        try:
            idx_bins = idx//s.chunk_capacity
        except: # idx may be array
            idx = np.asarray(idx)
            idx_bins = idx//s.chunk_capacity
        # idx_bins = np.floor_divide(idx,s.chunk_capacity)
        s.chunk_modified[idx_bins] = True
        s.arr[idx] = value
        
    def save(s,dir_path):
        """ save data to directory path dir_path incrementally"""
        dir_path = os.path.abspath(dir_path) # convert to absolute path
        try:os.mkdir(dir_path)
        except OSError:pass
#         print(f"saving data to {dir_path}")
        # remove previously generated chunk file
        for p in os.scandir(dir_path):
            try:
                k = int(p.name) # convert name to int
                if not s.chunk_saved[k]: # file not in chunk_saved
                    os.remove(p.path)
            except IndexError: # file not in chunk_saved
                os.remove(p.path)
            except ValueError: # not [0-9] file
                pass

        s.chunk_saved +=s.chunk_modified # update saved
        for k in np.flatnonzero(s.chunk_modified):
            s._saveArr(os.path.join(dir_path,f"{k}"),
                       s.arr[s.chunk_bins[k]:s.chunk_bins[k+1]])
        with open(os.path.join(dir_path,"header.msgpack"), 'wb') as file:
            msgpack.pack((s.capacity,tuple(s.shape),f"{s.arr.dtype}",s.chunk_bytes,
                          s.chunk_saved.tolist()), file)
        s.chunk_modified[:] = False # reset chunk_modified

    def load(s,dir_path):
        """load data from directory path dir_path"""
        dir_path = os.path.abspath(dir_path) # convert to absolute path
        with open(os.path.join(dir_path,"header.msgpack"), 'rb') as file:
            capacity,shape,dtype,chunk_bytes,chunk_saved = msgpack.unpack(file)  # restore
        s._init_arr(capacity,shape,dtype,chunk_bytes) # init arr
        s.chunk_saved = np.asarray(chunk_saved,dtype=bool) # restore chunk_saved
        for p in os.scandir(dir_path): # load chunks from iterator of os.DirEntry objects
            try:
                k = int(p.name) # convert name to int
                if s.chunk_saved[k]: # file in chunk_saved, load it
                    idx_start = s.chunk_bins[k]
                    idx_end = s.chunk_bins[k+1]
                    s.arr[idx_start:idx_end] = s._loadArr(p.path,dtype,(idx_end-idx_start,*shape))
            except ValueError: # not [0-9] file
                pass
            except IndexError: # file not in chunk_saved
                pass

    @staticmethod    
    def _saveArr(path,arr):
        """ helper function to save a single numpy arr to path, no header is written"""
        with open(path, 'wb+') as fh:
            fh.write(arr.data) # write raw buffer to file
    @staticmethod
    def _loadArr(path,dtype,shape):
        """ helper function to load a single numpy arr (buffer) from path"""
        with open(path, 'rb') as fh:
            return np.fromfile(fh, dtype=dtype).reshape(shape)



class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(s, # self
                 capacity: int,  # total number of sample
                 obs_shape: tuple,  # observation shape
                 action_shape: tuple,  # action shape
                 obs_dtype: type = np.float32,  # observation dtype
                 action_dtype: type = np.float32,  # action dtype
                 device="cpu",  # cpu, cuda, cuda:0, etc.
                 chunk_bytes = 2e7 # num bytes per chunk
                 ):
        s.capacity = capacity = int(capacity)
        s.device = device

        s.idx = 0
        s.full = False
        s.num_recent = -1  # int, most recent samples for updating
        #print(capacity,obs_shape,obs_dtype,chunk_bytes)
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        #obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8        
        s.obses =      RecordArray(capacity,obs_shape,obs_dtype,chunk_bytes)
        s.next_obses = RecordArray(capacity,obs_shape,obs_dtype,chunk_bytes)
        s.actions =    RecordArray(capacity,action_shape,action_dtype,chunk_bytes)
        s.rewards =    RecordArray(capacity,(1,),np.float32,chunk_bytes)
        s.not_dones =  RecordArray(capacity,(1,),np.float32,chunk_bytes)
        s.not_dones_no_max = RecordArray(capacity,(1,),np.float32,chunk_bytes)

    def __len__(s):
        return s.capacity if s.full else s.idx

    # def add(s, obs, action, reward, next_obs, done, done_no_max):
    def add(s, obs, action, reward, next_obs, not_done, not_done_no_max):
        """ add one sample """
        s.obses[s.idx] = obs
        s.actions[s.idx] = action
        s.rewards[s.idx] = reward
        s.next_obses[s.idx] = next_obs
        # s.not_dones[s.idx] = not done
        # s.not_dones_no_max[s.idx] = not done_no_max
        s.not_dones[s.idx] = not_done
        s.not_dones_no_max[s.idx] = not_done_no_max
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
        if not s.full:
            idx_begin = max(idx_begin, 0)
            
        #print(idx_begin, idx_end, batch_size)
        idxs = np.random.randint(idx_begin, idx_end, size=batch_size)
        obses = torch.as_tensor(s.obses[idxs], device=s.device)
        actions = torch.as_tensor(s.actions[idxs], device=s.device)
        rewards = torch.as_tensor(s.rewards[idxs], device=s.device)
        next_obses = torch.as_tensor(s.next_obses[idxs], device=s.device)
        not_dones = torch.as_tensor(s.not_dones[idxs], device=s.device)
        not_dones_no_max = torch.as_tensor(s.not_dones_no_max[idxs], device=s.device)
        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max

    def __sizeof__(s):
        """to estimate the size in memory in bytes"""
        return super().__sizeof__()+s.obses.__sizeof__() +\
            s.next_obses.__sizeof__()+s.actions.__sizeof__() +\
            s.rewards.__sizeof__()+s.not_dones.__sizeof__() + \
            s.not_dones_no_max.__sizeof__()

    def save(s,dir_path):
        """ save replay buffer to directory path dir_path"""
        dir_path = os.path.abspath(dir_path) # convert to absolute path
        try:os.mkdir(dir_path)
        except OSError:pass
        print(f"saving replay buffer to {dir_path}")
        s.obses.save(os.path.join(dir_path,"obses"))
        s.next_obses.save(os.path.join(dir_path,"next_obses"))
        s.actions.save(os.path.join(dir_path,"actions"))
        s.rewards.save(os.path.join(dir_path,"rewards"))
        s.not_dones.save(os.path.join(dir_path,"not_dones"))
        s.not_dones_no_max.save(os.path.join(dir_path,"not_dones_no_max"))
        with open(os.path.join(dir_path,"header.msgpack"), 'wb') as file:
            msgpack.pack((s.capacity,s.idx,s.full), file)

    def load(s,dir_path):
        """load replay buffer from directory path dir_path"""
        dir_path = os.path.abspath(dir_path) # convert to absolute path
        s.obses.load(os.path.join(dir_path,"obses"))
        s.next_obses.load(os.path.join(dir_path,"next_obses"))
        s.actions.load(os.path.join(dir_path,"actions"))
        s.rewards.load(os.path.join(dir_path,"rewards"))
        s.not_dones.load(os.path.join(dir_path,"not_dones"))
        s.not_dones_no_max.load(os.path.join(dir_path,"not_dones_no_max"))
        with open(os.path.join(dir_path,"header.msgpack"), 'rb') as file:
            s.capacity,s.idx,s.full = msgpack.unpack(file)