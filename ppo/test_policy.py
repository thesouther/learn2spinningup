import time
import joblib
import os
import os.path as osp
from joblib import logger
import torch
from tools.utils import EpochLogger


def load_policy_and_env(fpath, itr="last", deterministic=False, devices=None):
    """
    载入环境和模型, 只用pytorch模型,没有tf实现
    """
    saves = []
    backend = "pytorch"
    if itr == "last":
        pytsave_path = osp.join(fpath, 'pyt_save')
        saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x) > 8 and 'model' in x]
        itr = "%d" % max(saves) if len(saves) > 0 else ""
    else:
        assert isinstance(itr, int), "Bad value provided for itr (needs to be int or 'last')."
        itr = "%d" % itr
    get_action = load_pytorch_policy(fpath, itr, deterministic, devices)
    try:
        state = joblib.load(osp.join(fpath, 'vars' + itr + '.pkl'))
        env = state["env"]
        # env.reset()
    except:
        env = None
    return env, get_action


def load_pytorch_policy(fpath, itr, deterministic=False, devices=None):
    fname = osp.join(fpath, "pyt_save", "model" + itr + ".pt")
    print("\n\nloading from %s.\n\n" % fname)
    model = torch.load(fname)

    def get_action(x):
        """
        网络输出, 这里的action得减一维
        """
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0).to(devices)
            action = model.act(x)
        return action[0]

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."
    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print("Episode %d \t EpRet %.3f \t EpLen %d" % (n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1
    logger.log_tabular("EpRet", with_min_and_max=True)
    logger.log_tabular("EpLen", average_only=True)
    logger.dump_tabular()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("fpath", type=str)
    parser.add_argument('--len', '-l', type=int, default=1000)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--use_gpu', '-ug', action='store_true')
    args = parser.parse_args()

    if args.use_gpu:
        devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        devices = torch.device("cpu")

    env, get_action = load_policy_and_env(args.fpath,
                                          args.itr if args.itr >= 0 else "last",
                                          args.deterministic,
                                          devices=devices)
    run_policy(env, get_action, args.len, args.episodes, not (args.norender))
