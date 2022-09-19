import torch
import torch.nn as nn
import torch.nn.functional as F
'''
使用networkx编码,构造读取每个特征的函数,构造读取全部特征的函数
对uav编码, 即1-100, 构建节点信息。         shape: [n, 3]
or 对uav以及存在的包进行编码, 构建节点信息。  shape: [n, 51]  --
-- uav feature: 3, packet feature: 2, max packet num: 48 / 2, if not, fill 0
对packet编码, 即1-100, 处于某个uav节点     shape: [m, 3] 
-- 3: uav, feature1, feature2
对link编码, 即1-100, 连接哪两个节点         shape: [k, 4]
-- 4: src_uav, dst_uav, feature1, feature2 
'''

'''
GNN: 
input:  [n, uav_feature+packet_feature]->[n, 5]  graph: [link_feature, n, n]->[2, n, n]
output: [n, 1] 每个节点特征的权重
'''

class Encoder(nn.Module):
	def __init__(self, din, hidden_dim=32):
		super(Encoder, self).__init__()
		self.fc = nn.Linear(din, hidden_dim)

	def forward(self, x):
		embedding = F.relu(self.fc(x))
		return embedding

if __name__ == "__main__":
    # x = torch.ones(4, 5)  # (4, 32)
    x = torch.ones(5, 4, 4) # (4, 4, 32)
    net = Encoder(5, 32)
    out = net(x.permute(1,2,0))
    print(out.shape)