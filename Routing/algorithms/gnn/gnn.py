import sys
sys.path.append('/Users/ziping.yu/Documents/RL_DRONETWORK')
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/home/ziping.yu/work/pytorch_DGN/Routing')
from algorithms.gnn.layers import EdgeGraphAttentionLayer
from algorithms.gnn.layers import GraphAttentionLayer, SpGraphAttentionLayer

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

'''
TODO: GNN最后的output是什么, 即能学习到什么
      input:  packet(n, k1)、link(m, k2)
      output: ?
'''

class GAT(nn.Module):
    '''
    ref from https://github.com/Diego999/pyGAT
    '''
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

class EGAT(nn.Module):
    def __init__(self, nfeat, ef_sz, nhid, nclass, dropout, alpha, nheads):
        """
        ref: https://github.com/jamesYu365/EGAT
        Dense version of GAT.
        nfeat输入节点的特征向量长度, 标量
        ef_sz输入edge特征矩阵的大小, 列表, PxNxN
        nhid隐藏节点的特征向量长度, 标量
        nclass输出节点的特征向量长度, 标量
        dropout: drpout的概率
        alpha: leakyrelu的第三象限斜率
        nheads: attention_head的个数
        """
        super(EGAT, self).__init__()
        self.dropout = dropout
        self.nfeat = nfeat
        # 起始层
        self.attentions = [EdgeGraphAttentionLayer(nfeat, nhid[0], dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads[0])]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        
        # #hidden层
        # self.hidden_atts=[EdgeGraphAttentionLayer(nhid[0]*nheads[0]*ef_sz[0], nhid[1], dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads[1])]
        # for i, attention in enumerate(self.hidden_atts):
        #     self.add_module('hidden_att_{}'.format(i), attention)
        
        #输出层
        self.out_att = EdgeGraphAttentionLayer(nhid[0]*nheads[0]*ef_sz[0], nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, edge_attr):
        #起始层
        x = F.dropout(x, self.dropout, training=self.training)#起始层
        temp_x=[]
        for att in self.attentions:
            inn_x, edge_attr=att(x, edge_attr)
            temp_x.append(inn_x)
        x = torch.cat(temp_x, dim=1) # 起始层[10, 128]
        
        
        # #中间层
        # x = F.dropout(x, self.dropout, training=self.training)#中间层
        # temp_x=[]
        # for att in self.hidden_atts:
        #     inn_x,edge_attr=att(x, edge_attr)
        #     temp_x.append(inn_x)
        # x = torch.cat(temp_x, dim=1)#中间层
        
        
        #输出层
        x = F.dropout(x, self.dropout, training=self.training)  # 输出层[10, 128]
        x = self.out_att(x, edge_attr)    # x.shape [N, out_features]
        x = F.elu(x)  # 输出层[N, out_features]
        return F.log_softmax(x, dim=1) # [N, out_features]

if __name__ == "__main__":
    args={
    'no_cuda':True,
    #'no_cuda':False,
    'fastmode':False,
    'seed':72,
    'epochs':10000,
    'lr':0.005,
    'weight_decay':5e-4,
    'hidden':[64,8],
    'nb_heads':[1,8],
    'dropout':0.6,
    'alpha':0.2,
    'patience':50,
    'batch_size':20
    }
    # (nodes, features)
    x = torch.ones(10, 2)
    edge_attr = torch.ones(2, 10, 10)
    egat = EGAT(nfeat=x.shape[1], ef_sz=tuple(edge_attr.shape), nhid=args['hidden'], 
                nclass=x.shape[1], dropout=args['dropout'], alpha=args['alpha'], nheads=args['nb_heads'])
    out = egat(x, edge_attr)
    print(out.shape)