import torch
from torch.optim import Adam
import numpy as np
from textwrap import dedent
import gym
from copy import deepcopy
import time
from functools import reduce

from tools.utils import setup_logger_kwargs, ReplayBuffer, EpochLogger
from model import MLPActorCritic, count_vars
from tools.config import devices


# def ddpg(env_fn,
#          actor_critc=MLPActorCritic,
#          ac_kwargs=dict(),
#          seed=0,
#          steps_per_epoch=4000,
#          epochs=100,
#          replay_size=int(1e4),
#          gamma=0.99,
#          polyak=0.995,
#          pi_lr=1e-3,
#          q_lr=1e-3,
#          batch_size=100,
#          start_steps=10000,
#          update_after=1000,
#          update_every=50,
#          act_noise=0.1,
#          num_test_episodes=10,
#          max_ep_len=1000,
#          logger_kwargs=dict(),
#          save_freq=1):
def ddpg(env_fn,
         actor_critc=MLPActorCritic,
         ac_kwargs=dict(),
         seed=0,
         steps_per_epoch=50,
         epochs=5,
         replay_size=int(1e4),
         gamma=0.99,
         polyak=0.995,
         pi_lr=1e-3,
         q_lr=1e-3,
         batch_size=10,
         start_steps=10,
         update_after=10,
         update_every=5,
         act_noise=0.1,
         num_test_episodes=10,
         max_ep_len=50,
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
    act_limit = env.action_space.high[0]
    # print(obs_dim)
    # print(act_dim)
    # print(act_limit)
    ac = actor_critc(env.observation_space, env.action_space, **ac_kwargs).to(devices)
    ac_targ = deepcopy(ac).to(devices)

    for p in ac_targ.parameters():
        p.requires_grad = False

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    # print("replay_buffer.obs_buf.shape", replay_buffer.obs_buf.shape)
    # print("replay_buffer.act_buf.shape", replay_buffer.act_buf.shape)
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.q])
    logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n' % var_counts)

    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'].to(devices), data['act'].to(devices), \
            data['rew'].to(devices), data['obs2'].to(devices), data['done'].to(devices)
        # 这里有一些环境的观察是图片， 所以把观察展平成向量
        # print("compute_loss_q o.shape,a.shape: ", o.size(), a.shape)
        q = ac.q(o, a)
        with torch.no_grad():
            a2 = ac_targ.pi(o2).to(devices)
            # print("compute_loss_q a2 shape", a2.shape)
            q_pi_targ = ac_targ.q(o2, a2)
            backup = r + gamma * (1 - d) * q_pi_targ
        loss_q = ((q - backup)**2).mean()
        loss_info = dict(QVals=q.detach().cpu().numpy()) if devices.type=='cuda' \
            else dict(QVals=q.detach().numpy())
        return loss_q, loss_info

    def compute_loss_pi(data):
        o = data['obs'].to(devices)
        a = ac.pi(o).to(devices)
        q_pi = ac.q(o, a)
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
        # print("loss_q, loss_info", loss_q, loss_info)

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

        logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # 最后软更新目标网络参数
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        """
        网络输出, 这里的action得减一维
        """
        o = torch.as_tensor(o.copy(), dtype=torch.float32).unsqueeze(0).to(devices)
        a = ac.act(o)
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)[0]

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

    for t in range(total_steps):
        if t > start_steps:
            a = get_action(o, act_noise)  # (3,)
        else:
            a = env.action_space.sample()  #(3,)

        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        d = False if ep_len == max_ep_len else d
        replay_buffer.store(o, a, r, o2, d)
        o = o2
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)
            test_agent()

            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default='train', help="[train, plot, test]")
    parser.add_argument('--env', type=str, default='CarRacing-v0', help="环境名称")
    parser.add_argument('--hid', type=list, default=[128, 128], help="模型隐藏层神经元数量")
    parser.add_argument('--l', type=int, default=2, help="模型层数")
    parser.add_argument('--gamma', type=float, default=0.99, help="折扣因子")
    parser.add_argument('--seed', type=int, default=0, help="随机数种子")
    parser.add_argument('--epochs', type=int, default=5, help="回合数")
    parser.add_argument('--exp_name', type=str, default='ddpg', help="实验名称")
    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ddpg(lambda: gym.make(args.env),
         actor_critc=MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=args.hid),
         gamma=args.gamma,
         seed=args.seed,
         epochs=args.epochs,
         logger_kwargs=logger_kwargs)
