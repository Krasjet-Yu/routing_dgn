import math
import random
import numpy as np
import torch
from torch import float32
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.append('/home/ziping.yu/work/pytorch_DGN/Routing')
from config import Config
from buffer import ReplayBuffer
from algorithms.encoder.encode import Encoder

class AttModel(nn.Module):
	def __init__(self, n_node, din, hidden_dim, dout):
		super(AttModel, self).__init__()
		self.fcv = nn.Linear(din, hidden_dim)
		self.fck = nn.Linear(din, hidden_dim)
		self.fcq = nn.Linear(din, hidden_dim)
		self.fcout = nn.Linear(hidden_dim, dout)

	def forward(self, x, mask):
		v = F.relu(self.fcv(x))
		q = F.relu(self.fcq(x))
		k = F.relu(self.fck(x)).permute(0,2,1)
		att = F.softmax(torch.mul(torch.bmm(q,k), mask) - 9e15*(1 - mask),dim=2)

		out = torch.bmm(att,v)
		#out = torch.add(out,v)
		out = F.relu(self.fcout(out))
		return out

class Q_Net(nn.Module):
	def __init__(self, hidden_dim, dout):
		super(Q_Net, self).__init__()
		self.fc = nn.Linear(hidden_dim, dout)

	def forward(self, x):
		q = self.fc(x)
		return q

class DGN(nn.Module):
	def __init__(self,n_agent,num_inputs,hidden_dim,num_actions):
		super(DGN, self).__init__()
		
		self.encoder = Encoder(num_inputs,hidden_dim)
		self.att_1 = AttModel(n_agent,hidden_dim,hidden_dim,hidden_dim)
		self.att_2 = AttModel(n_agent,hidden_dim,hidden_dim,hidden_dim)
		self.q_net = Q_Net(hidden_dim,num_actions)
		
	def forward(self, x, mask):
		h1 = self.encoder(x)
		h2 = self.att_1(h1, mask)
		h3 = self.att_2(h2, mask)
		q = self.q_net(h3)
		return q 

class MDGN:
	def __init__(self, n_agent, n_inputs, n_actions, cfg):
		self.n_actions = n_actions  # 总的动作个数
		self.n_agent = n_agent
		self.device = cfg.device
		self.gamma = cfg.gamma  # 奖励的折扣因子
		# e-greedy策略相关参数
		self.frame_idx = 0  # 用于epsilon的衰减计数
		self.epsilon = lambda frame_idx: cfg.epsilon_end + \
			(cfg.epsilon_start - cfg.epsilon_end) * \
			math.exp(-1. * frame_idx / cfg.epsilon_decay)
		self.batch_size = cfg.batch_size
		self.policy_net = DGN(n_agent, n_inputs, cfg.hidden_dim, n_actions).to(self.device)
		self.target_net = DGN(n_agent, n_inputs, cfg.hidden_dim, n_actions).to(self.device)
		self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
		self.memory = ReplayBuffer(cfg.memory_capacity)
    
	def choose_action(self, state, adj):
		''' choose action
		'''
		self.frame_idx += 1
		actions=[]
		with torch.no_grad():
			state = torch.tensor(np.array([state]), device=self.device, dtype=torch.float32)
			adj = torch.tensor(adj, device=self.device, dtype=torch.float32)
			q_values = self.policy_net(state, adj)[0]  # torch.Size([1, 20, 5]) --> torch.Size([20, 5])
			for i in range(self.n_agent):
				if random.random() < self.epsilon(self.frame_idx):
					a = np.random.randint(self.n_actions)
				else:
					a = q_values[i].argmax().item()
				actions.append(a)
		return actions

	def update(self):
		if len(self.memory) < self.batch_size: # 当memory中不满足一个批量时，不更新策略
			return 0
		state_batch, action_batch, reward_batch, next_state_batch, adj_batch, next_adj_batch, done_batch = self.memory.sample(
			self.batch_size)
		state_batch      = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float)
		# action_batch     = torch.tensor(action_batch, device=self.device) 
		# reward_batch     = torch.tensor(reward_batch, device=self.device, dtype=torch.float)  
		next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float)
		adj_batch        = torch.tensor(np.squeeze(adj_batch, 1), device=self.device, dtype=torch.float)  
		next_adj_batch   = torch.tensor(np.squeeze(next_adj_batch, 1), device=self.device, dtype=torch.float)  
		# done_batch       = torch.tensor(np.float32(done_batch), device=self.device)
		q_values = self.policy_net(state_batch, adj_batch)
		next_q_values = self.target_net(next_state_batch, next_adj_batch).max(dim = 2)[0]
  
		next_q_values = np.array(next_q_values.cpu().data)
		expected_q = np.array(q_values.cpu().data)
		for j in range(self.batch_size):
			for i in range(self.n_actions):
				expected_q[j][i][action_batch[j][i]] = reward_batch[j][i] + (1-done_batch[j][i])*self.gamma*next_q_values[j][i]
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
    n_agents = 20
    n_states = 45
    n_actions = 5
    state = np.ones((n_agents, n_states))
    # state = torch.tensor(state, dtype=float32).cuda()
    adj   = np.ones((1,n_agents,n_agents))
    # adj   = torch.tensor(adj, dtype=float32).cuda() 
    agent = MDGN(n_agents, n_states, n_actions, cfg) # 创建智能体
    action = agent.choose_action(state, adj)
    print(action)