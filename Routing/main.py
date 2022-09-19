import numpy as np
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter   
from common.utils import save_results_1, make_dir
from common.utils import plot_rewards,save_args,plot_losses

from algorithms.my_algorithms.tegan_rl import TEGAN_DQN
from routing import Routing
from config import Config

def env_agent_config(cfg):
    ''' 创建环境和智能体
    '''
    env = Routing(n_agent=cfg.uav, n_neighbor=cfg.neighbor, n_data=cfg.active_nodes) # 创建环境
    n_states = env.n_obs  # 状态维度
    n_agents = env.n_router
    n_actions = env.n_neighbor + 1
    agent = TEGAN_DQN(n_states, n_agents, n_actions, cfg)  # 创建智能体
    if cfg.seed !=0: # 设置随机种子
        torch.manual_seed(cfg.seed)
        env.seed(cfg.seed)
        np.random.seed(cfg.seed)
    return env, agent

def train(cfg, env, agent):
    ''' Training
    '''
    print('Start training!')
    print(f'Env: {cfg.env_name}, Algorithm: {cfg.algo_name}, device: {cfg.device}')
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    total_loss = []  
    steps = []
    writer = SummaryWriter(cfg.log_path)
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录一回合内的奖励
        ep_loss   = 0  # 记录一回合内的损失
        ep_step = 0
        state, adj = env.reset()  # 重置环境，返回初始状态
        for _ in tqdm(range(cfg.max_step), desc=f"Epochs{i_ep}/{cfg.train_eps}: "):
            ep_step += 1
            action = agent.choose_action(state, adj)  # 选择动作
            next_state, next_adj, reward, done = env.step(action)  # 更新环境，返回transition
            agent.memory.push(state, action, reward,
                              next_state, adj, next_adj, done)  # 保存transition
            state = next_state  # 更新下一个状态
            adj   = next_adj
            loss = agent.update()  # 更新智能体
            ep_reward += sum(reward)  # 累加奖励
            ep_loss   += loss
        
        # ep_reward = ep_reward / 2000.0
        # ep_loss   = ep_loss   / cfg.max_step
        
        writer.add_scalar('Loss', ep_loss, i_ep)
        writer.add_scalar('Reward', ep_reward, i_ep)
        # writer.add_scalar('Loss', ep_loss, i_ep)
        
        if (i_ep + 1) % cfg.target_update == 0:  # 智能体目标网络更新
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        steps.append(ep_step)
        rewards.append(ep_reward)
        total_loss.append(ep_loss)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % 1 == 0:
            print(f'Episode:{i_ep+1}/{cfg.train_eps}, Loss:{ep_loss:.2f}, Reward:{ep_reward:.2f}, Step:{ep_step:.2f} Epislon:{agent.epsilon(agent.frame_idx):.3f}')
    print('Finish training!')
    # env.close()
    res_dic = {'rewards':rewards,'ma_rewards':ma_rewards,'loss':total_loss, 'steps':steps}
    return res_dic


def test(cfg, env, agent):
    print('Start Test!')
    print(f'Env: {cfg.env_name}, Algorithm: {cfg.algo_name}, device: {cfg.device}')
    ############# 由于测试不需要使用epsilon-greedy策略，所以相应的值设置为0 ###############
    cfg.epsilon_start = 0.0  # e-greedy策略中初始epsilon
    cfg.epsilon_end = 0.0  # e-greedy策略中的终止epsilon
    ################################################################################
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    steps = []
    for i_ep in tqdm(range(cfg.test_eps), desc="Testing"):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state, adj = env.reset()  # 重置环境，返回初始状态
        for _ in range(cfg.max_step):
            ep_step+=1
            action = agent.choose_action(state, adj)  # 选择动作
            next_state, next_adj, reward, done = env.step(action)  # 更新环境，返回transition
            state = next_state  # 更新下一个状态
            adj   = next_adj 
            ep_reward += sum(reward)  # 累加奖励


        steps.append(ep_step)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f'Episode:{i_ep+1}/{cfg.test_eps}, Reward:{ep_reward:.2f}, Step:{ep_step:.2f}')
    print('Finish testing')
    # env.close()
    return {'rewards':rewards,'ma_rewards':ma_rewards,'steps':steps}


if __name__ == "__main__":
    cfg = Config()
    # 训练
    env, agent = env_agent_config(cfg)
    res_dic = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)  # 创建保存结果和模型路径的文件夹
    save_args(cfg)
    agent.save(path=cfg.model_path)  # 保存模型
    save_results_1(res_dic, tag='train',
                 path=cfg.result_path)  # 保存结果
    plot_rewards(res_dic['rewards'], res_dic['ma_rewards'], cfg, tag="train")  # 画出结果
    plot_losses(res_dic['loss'], path=cfg.result_path)
    # 测试
    env, agent = env_agent_config(cfg)
    agent.load(path=cfg.model_path)  # 导入模型
    res_dic = test(cfg, env, agent)
    save_results_1(res_dic, tag='test',
                 path=cfg.result_path)  # 保存结果
    plot_rewards(res_dic['rewards'], res_dic['ma_rewards'],cfg, tag="test")  # 画出结果




		