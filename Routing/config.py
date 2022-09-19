import os, sys
curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path) # 父路径
sys.path.append(parent_path) # 添加路径到系统路径
import datetime
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
import torch

class Config:
    def __init__(self) -> None:
        ################################## 环境超参数 ###################################
        self.algo_name = "DGN"  # 算法名称
        self.env_name = 'Routing' # 环境名称
        self.continuous = False # 环境是否为连续动作
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.seed = 476 # 随机种子，置0则不设置随机种子
        self.max_step = 200 # 200
        self.train_eps = 500 # 训练的回合数 500
        self.test_eps = 100 # 测试的回合数
        ################################################################################
        
        ################################## 算法超参数 ####################################
        self.uav            = 20
        self.active_nodes   = 20   # 数据包的数量
        self.node_features  = 5    # 数据包的特征
        self.edge_num       = 20   # 拓扑边的数量
        self.edge_features  = 5    # 拓扑边的特征
        self.neighbor       = 4    # 邻居的最大数量
        
        self.encoder_hidden = 128      # encoder隐藏层特征
        self.gnn_hidden     = [64,8]   # gnn隐藏层特征
        self.gnn_dropout    = 0.6      # gnn的dropout
        self.alpha          = 0.2      # gnn超参数
        self.heads          = [1,8]
        
        self.atten_hidden   = 128 
        
        self.out_channels   = [128, 64, 64, 128]      # TemporalConvNet
        self.kernel_size    = 2        # 空洞卷积核
        self.seq            = 4
        self.capacity       = 10
        self.t_dropout      = 0.2
        self.operator       = "cut"
        self.single         = 1
        self.decoder_hidden = 128  
        
        self.batch_size = 128  # mini-batch SGD中的批量大小 128
        self.gamma = 0.98  # 强化学习中的折扣因子
        self.epsilon_start = 0.95
        self.epsilon_end = 0.1
        self.n_epochs = 5
        self.epsilon_decay = 50
        self.lr = 0.0003
        self.target_update = 4
        self.actor_lr = 0.0003 # actor的学习率
        self.critic_lr = 0.0003 # critic的学习率
        self.gae_lambda = 0.95
        self.policy_clip = 0.2
        self.hidden_dim = 128
        self.update_fre = 20 # 策略更新频率
        self.memory_capacity = 10000  # replay buffer 10000
        ################################################################################
        
        ################################# 保存结果相关参数 ################################
        self.result_path = curr_path+"/outputs/" + self.env_name + \
            '/'+curr_time+'/results/'  # 保存结果的路径
        self.model_path = curr_path+"/outputs/" + self.env_name + \
            '/'+curr_time+'/models/'  # 保存模型的路径
        self.log_path   = curr_path+"/outputs/" + self.env_name + \
            '/'+curr_time+'/logs/'  # 保存log的路径
        self.save = True # 是否保存图片
        self.save_fig = True # 是否保存配置
        ################################################################################
