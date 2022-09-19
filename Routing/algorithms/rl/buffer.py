import random
import collections

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity # 经验回放的容量
        self.buffer = [] # 缓冲区
        self.position = 0 
    
    def push(self, state, action, reward, next_state, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity 
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # 随机采出小批量转移
        state, action, reward, next_state, done =  zip(*batch) # 解压成状态，动作等
        return state, action, reward, next_state, done
    
    def __len__(self):
        ''' 返回当前存储的量
        '''
        return len(self.buffer)
    
class TReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity # 经验回放的容量
        self.buffer = [] # 缓冲区
        self.position = 0 
    
    def push(self, state, action, reward, next_state, adj, next_adj, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, adj, next_adj, done)
        self.position = (self.position + 1) % self.capacity 
    
    def sample(self, batch_size):
        pos = random.randint(batch_size, len(self.buffer))
        batch = self.buffer[pos-batch_size: pos] # 随机采出小批量转移
        state, action, reward, next_state, adj, next_adj, done =  zip(*batch) # 解压成状态，动作等
        return state, action, reward, next_state, adj, next_adj, done
    
    def __len__(self):
        ''' 返回当前存储的量
        '''
        return len(self.buffer)

class TBuffer:
    def __init__(self, capacity=4):
        # self.capacity = capacity # 时间序列的容量
        self.buffer = collections.deque(maxlen=capacity) # 缓冲区
    def push(self, state, adj):
        ''' 
        缓冲区是一个双向队列，容量超出时去掉开始存入的转移(transition)
        '''
        self.buffer.append((state, adj))
    
    def sample(self):
        batch = [item for item in self.buffer]
        # batch = torch.tensor(self.buffer, dtype=torch.float32).cuda()
        state, adj = zip(*batch)
        return state, adj
    
    def __len__(self):
        ''' 
        返回当前存储的量
        '''
        return len(self.buffer)