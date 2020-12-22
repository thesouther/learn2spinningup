import torch
from torch.optim import Adam
import numpy as np
from textwrap import dedent
import gym
from copy import deepcopy
import time
from utils import setup_logger_kwargs, ReplayBuffer, EpochLogger
from model import MLPActorCritic, count_vars


def ddpg(env_fn,
         actor_critc=MLPActorCritic,
         ac_kwargs=dict(),
         seed=0,
         steps_per_epoch=4000,
         epochs=100,
         replay_size=int(1e4),
         gamma=0.99,
         polyak=0.995,
         pi_lr=1e-3,
         q_lr=1e-3,
         batch_size=100,
         start_steps=10000,
         update_after=1000,
         update_every=50,
         act_noise=0.1,
         num_test_episodes=10,
         max_ep_len=1000,
         logger_kwargs=dict(),
         save_freq=1):
    """
    主要参数：
        polyak(float): 参数软更新时的\\rho
    """
    # 实例化log
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    #设置随机数种子
    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    print(obs_dim)
    print(act_dim)
    act_limit = env.action_space.high[0]
    print(act_limit)
    ac = actor_critc(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    for p in ac_targ.parameters():
        p.requires_grad = False

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    print(replay_buffer.obs_buf.shape)
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.q])
    logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n' % var_counts)

    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q = ac.q(o, a)
        with torch.no_grad():
            q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
            backup = r + gamma * (1 - d) * q_pi_targ
        loss_q = ((q - backup)**2).mean()
        loss_info = dict(QVals=q.detach().numpy())
        return loss_q, loss_info

    def compute_loss_pi(data):
        o = data['obs']
        q_pi = ac.q(o, ac.pi(o))
        return -q_pi.mean()

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(ac.q.parameters(), lr=q_lr)
    logger.setup_pytorch_saver(ac)

    def update(data):
        # 首先对Q执行一步梯度下降
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # 冻结Q网络， 提高计算效率
        for p in ac.q.parameters():
            p.requires_grad = False

        # 对pi执行一步梯度下降
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # 解冻Q参数
        for p in ac.q.parameters():
            p.requires_grad = True

        logger.store(LossQ=loss_q.items(), LossPi=loss_pi.items(), **loss_info)

        # 最后软更新目标网络参数
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CarRacing-v0', help="环境名称")
    parser.add_argument('--hid', type=int, default=256, help="模型隐藏层神经元数量")
    parser.add_argument('--l', type=int, default=2, help="模型层数")
    parser.add_argument('--gamma', type=float, default=0.99, help="折扣因子")
    parser.add_argument('--seed', type=int, default=0, help="随机数种子")
    parser.add_argument('--epochs', type=int, default=50, help="回合数")
    parser.add_argument('--exp_name', type=str, default='ddpg', help="实验名称")
    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    ddpg(lambda: gym.make(args.env),
         actor_critc=MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
         gamma=args.gamma,
         seed=args.seed,
         epochs=args.epochs,
         logger_kwargs=logger_kwargs)
