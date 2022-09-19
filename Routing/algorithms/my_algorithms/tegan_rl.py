'''Edge-Graph Attention and Temporal Network and Reinforcement Learning'''
import torch
import torch.nn as nn
import collections
# import pfrl
import random
import math
import torch.optim as optim
import numpy as np
import sys
sys.path.append('/home/ziping.yu/work/pytorch_DGN/Routing')
from config import Config
# from src.utilities import utilities as util
# from src.routing_algorithms.BASE_routing import BASE_routing
from algorithms.gnn.gnn import EGAT, AttModel
from algorithms.rnn.tcn import TemporalConvNet
from algorithms.encoder.encode import Encoder
from algorithms.rl.dqn import MLP
from algorithms.rl.buffer import TReplayBuffer, TBuffer

'''
    p:  num of packets
    pf: num of packet's features
    u:  num of uavs
    l:  num of links
    uf: num of uav's features
    lf: num of link's features
    k:  sequence
'''

class TEGAN(nn.Module):
    def __init__(self, n_states, n_agents, n_actions, cfg):
        super(TEGAN, self).__init__()
        # ddpgcfg = DDPGConfig()
        self.single = cfg.single
        self.device = cfg.device
        self.capacity = cfg.capacity
        '''
        input:  [b, n, 5]
        output: [b, n, 32]
        '''
        self.encoder1 = Encoder(din=n_states, hidden_dim=cfg.encoder_hidden).to(self.device)
        # self.encoder2 = Encoder(din=cfg.edge_features, hidden_dim=cfg.encoder_hidden).to(self.device)
        
        '''
        input:  [b, n, nfeat],       [b, n, n]
        output: [b, n, nfeat_hidden]
        '''
        self.att_1 = AttModel(n_agents, cfg.atten_hidden, cfg.atten_hidden, cfg.atten_hidden).to(self.device)
        self.att_2 = AttModel(n_agents, cfg.atten_hidden, cfg.atten_hidden, cfg.atten_hidden).to(self.device)
        
        '''
        input:  [n, 32, b]
        output: [n, 64]
        '''
        self.tcn = TemporalConvNet(num_inputs=cfg.encoder_hidden, num_channels=cfg.out_channels, 
                              kernel_size=cfg.kernel_size, dropout=cfg.t_dropout, operator=cfg.operator).to(self.device)
        
        '''
        input:  [n, 64]
        output: [n, a]
        '''
        self.decoder = MLP(num_inputs=cfg.out_channels[-1], action_dim=n_actions, hidden_dim=cfg.decoder_hidden).to(self.device)


    def forward(self, node, edge):
        '''
            node: torch.Size([b, n, nfeat]
            edge: torch.Size([b, n, n]
            k:    time sequence (int)
        '''
        # res = node # res.shape: torch.Size([b, n, nfeat])node.shape
        out = self.encoder1(node)                                # out.shape: torch.Size([b, n, nfeat_hidden])
        out = self.att_1(out, edge)
        out = self.att_2(out, edge)
        out = self.tcn(out.permute(1,2,0))
        # out = torch.cat((out, res), 2)
        out = self.decoder(out)
        # return pfrl.action_value.DiscreteActionValue(out)
        # out.shape: [n, actions]
        return out

class TEGAN_DQN:
    def __init__(self, n_states, n_agents, n_actions, cfg):

        self.action_dim = cfg.active_nodes        # 总的动作个数
        self.device = cfg.device                  # 设备，cpu或gpu等
        self.gamma = cfg.gamma                    # 奖励的折扣因子
        self.n_actions = n_actions
        self.n_agent   = n_agents
        self.obs_state = n_states
        self.memory_capacity = cfg.memory_capacity
        self.k_seq = cfg.seq
        # e-greedy策略相关参数
        self.frame_idx = 0                        # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
            (cfg.epsilon_start - cfg.epsilon_end) * \
            math.exp(-1. * frame_idx / (cfg.epsilon_end+1e-6))
        self.batch_size = cfg.batch_size
        self.policy_net = TEGAN(n_states, n_agents, n_actions, cfg).to(self.device)
        self.target_net = TEGAN(n_states, n_agents, n_actions, cfg).to(self.device)
        for target_param, param in zip(self.target_net.parameters(),self.policy_net.parameters()): # 复制参数到目标网路targe_net
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr) # 优化器
        self.memory = TReplayBuffer(cfg.memory_capacity) # 经验回放
        self.tbuffer = TBuffer(cfg.seq)

    def choose_action(self, state, adj):
        ''' 
        choose action
        '''
        self.frame_idx += 1
        actions=[]
        if len(self.tbuffer) < self.k_seq:
            fill = self.k_seq - len(self.tbuffer) - 1
            for _ in range(fill):
                self.tbuffer.push(state, adj[0])
        self.tbuffer.push(state, adj[0])
        state, adj = self.tbuffer.sample()
        with torch.no_grad():
            state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32)
            adj = torch.tensor(adj, device=self.device, dtype=torch.float32)
            q_values = self.policy_net(state, adj)    # torch.Size([1, action])
            for i in range(self.n_agent):
                if random.random() < self.epsilon(self.frame_idx):
                    a = np.random.randint(self.n_actions)
                else:
                    a = q_values[i].argmax().item()
                actions.append(a)
        return actions
    
    def update(self):
        if len(self.memory) < self.memory_capacity: # 当memory中不满足一个批量时，不更新策略
            return 0
        # batch是否可以从每一个回合中取？时序卷积是否影响了马尔可夫性，即当前时刻只与前一时刻有关
        state_batch, action_batch, reward_batch, next_state_batch, adj_batch, next_adj_batch, done_batch = self.memory.sample(
            self.batch_size)
        state_batch      = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float)  
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float)
        adj_batch        = torch.tensor(np.squeeze(adj_batch, 1), device=self.device, dtype=torch.float)  
        next_adj_batch   = torch.tensor(np.squeeze(next_adj_batch, 1), device=self.device, dtype=torch.float)  
        # done_batch       = torch.tensor(np.float32(done_batch), device=self.device)
        q_values = self.policy_net(state_batch, adj_batch)
        # next_q_values = self.target_net(next_state_batch, next_adj_batch).max(dim = 2)[0]
        next_q_values = self.target_net(next_state_batch, next_adj_batch).max(dim = 1)[0]

        next_q_values = np.array(next_q_values.cpu().data)
        expected_q = np.array(q_values.cpu().data)
        last_pos = self.batch_size - 1
        for i in range(self.n_actions):
            expected_q[i][action_batch[last_pos][i]] = reward_batch[last_pos][i] + (1-done_batch[last_pos][i])*self.gamma*next_q_values[i]
        loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean()
        # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        # expected_q_values = reward_batch + self.gamma * next_q_values * (1-done_batch)
        # loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  # 计算均方根损失

        # 优化更新模型
        self.optimizer.zero_grad()  
        loss.backward()
        for param in self.policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss

    def save(self, path):
        torch.save(self.target_net.state_dict(), path+'dgn_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path+'dgn_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)

if __name__ == "__main__":
    cfg = Config()
    x = torch.ones(5, 20, 5).cuda()
    edge_attr = torch.ones(5, 20, 20).cuda()
    net = TEGAN(5, 20, 20, cfg=cfg).cuda()
    out = net(x, edge_attr)
    print(out.shape)   # torch.Size([1, 20])