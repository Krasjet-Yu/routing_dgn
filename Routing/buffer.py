from collections import deque 
import random
class ReplayBuffer(object):

	def __init__(self, buffer_size):
		self.buffer_size = buffer_size
		self.num_experiences = 0
		self.buffer = deque()

	def sample(self, batch_size):
		if self.num_experiences < batch_size:
			batch = random.sample(self.buffer, self.num_experiences)
		else:
			batch = random.sample(self.buffer, batch_size)
		state, action, reward, next_state, adj, next_adj, done =  zip(*batch) # 解压成状态，动作等
		return state, action, reward, next_state, adj, next_adj, done

	def push(self, obs, action, reward, new_obs, matrix, next_matrix, done):
		experience = (obs, action, reward, new_obs, matrix, next_matrix, done)
		if self.num_experiences < self.buffer_size:
			self.buffer.append(experience)
			self.num_experiences += 1
		else:
			self.buffer.popleft()
			self.buffer.append(experience)
	
	def __len__(self):
		return len(self.buffer)