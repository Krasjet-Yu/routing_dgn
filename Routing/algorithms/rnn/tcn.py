#!/usr/bin/env python
# coding=utf-8
'''
@Author: Krasjet_Yu
@Email: krasjet.ziping@gmail.com
@Date: 2022-06-04 20:25:52
@Discription: 
@Environment: python 3.8.9
'''

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import math
import numpy as np


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块, 裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长,一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  #  裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (n, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    '''
        paper: https://arxiv.org/pdf/1803.01271.pdf
        code:  https://github.com/locuslab/TCN
    '''
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, operator="cat"):
        """
        TCN,目前paper给出的TCN结构很好的支持每个时刻为一个数的情况,即sequence结构,
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int, 输入通道数
        :param num_channels: list,每层的hidden_channel数,例如[25,25,25,25]表示有4个隐层,每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        self.operator = operator
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i   # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i-1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN,一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels),
        这里把seq_len放在channels后面,把所有时间步的数据拼起来,当做Conv1d的输入尺寸,实现卷积跨时间步的操作，
        很巧妙的设计。
        
        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        :     or size of (Batch, output_channel, seq_len)
        """
        if self.operator == "sum":
            return torch.sum(self.network(x), dim=2)                   # out.shape (nodes, features)
        elif self.operator == "mean":
            return torch.mean(self.network(x), dim=2, keepdim=False)   # out.shape (nodes, features)
        elif self.operator == "cut":
            return self.network(x)[:, :, -1]                           # out.shape (nodes, features)
        else:
            return self.network(x)                                     # out.shaoe (nodes, features, seq_len)

class TAConvNet(nn.Module):
  def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, max_length=200, attention=False):
    super(TAConvNet, self).__init__()
    layers = []
    num_levels = len(num_channels)
    for i in range(num_levels):
      dilation_size = 2 ** i
      in_channels = num_inputs if i == 0 else num_channels[i-1]
      out_channels = num_channels[i]
      layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                               padding=(kernel_size-1) * dilation_size, dropout=dropout)]
      if attention == True:
        layers += [AttentionBlock(max_length, max_length, max_length)]

    self.network = nn.Sequential(*layers)

  def forward(self, x):
    return self.network(x)

class AttentionBlock(nn.Module):
  """An attention mechanism similar to Vaswani et al (2017)
  The input of the AttentionBlock is `BxTxD` where `B` is the input
  minibatch size, `T` is the length of the sequence `D` is the dimensions of
  each feature.
  The output of the AttentionBlock is `BxTx(D+V)` where `V` is the size of the
  attention values.
  Arguments:
      dims (int): the number of dimensions (or channels) of each element in
          the input sequence
      k_size (int): the size of the attention keys
      v_size (int): the size of the attention values
      seq_len (int): the length of the input and output sequences
  """
  def __init__(self, dims, k_size, v_size, seq_len=None):
    super(AttentionBlock, self).__init__()
    self.key_layer = nn.Linear(dims, k_size)
    self.query_layer = nn.Linear(dims, k_size)
    self.value_layer = nn.Linear(dims, v_size)
    self.sqrt_k = math.sqrt(k_size)

  def forward(self, minibatch):
    keys = self.key_layer(minibatch)
    queries = self.query_layer(minibatch)
    values = self.value_layer(minibatch)
    logits = torch.bmm(queries, keys.transpose(2,1))
    # Use numpy triu because you can't do 3D triu with PyTorch
    # TODO: using float32 here might break for non FloatTensor inputs.
    # Should update this later to use numpy/PyTorch types of the input.
    mask = np.triu(np.ones(logits.size()), k=1).astype('uint8')
    mask = torch.from_numpy(mask).cuda()
    # do masked_fill_ on data rather than Variable because PyTorch doesn't
    # support masked_fill_ w/-inf directly on Variables for some reason.
    logits.data.masked_fill_(mask, float('-inf'))
    probs = F.softmax(logits, dim=1) / self.sqrt_k
    read = torch.bmm(probs, values)
    return minibatch + read

if __name__ == '__main__':
    # batch, features, sequence_len
    args = {
        "nodes": 4,
        "features": 2,
        "seq_len": 5,
        "kernel_size": 2,
        "dropout": 0.2,
        "out_channels": [16, 32, 16, 2]
    }
    x = torch.ones(args['nodes'], args['features'], args['seq_len'])
    tcn = TemporalConvNet(num_inputs=x.shape[1], num_channels=args['out_channels'], kernel_size=args['kernel_size'], dropout=args['dropout'], operator="cut")
    out = tcn(x)  # [n, features]
    # out1 = torch.mean(out, dim=2, keepdim=False)
    print(f"out.shape: {out.shape}")
    # print(f"out1.shape: {out1.shape}")
    # print(f"out: {out}")
    # print(f"out[:, :, -1]: {out[:, :, -1]}")
    # print(f"out mean: {out1}")