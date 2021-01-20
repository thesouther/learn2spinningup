import subprocess
import sys
import gym
from textwrap import dedent
from copy import deepcopy
from numpy.lib.shape_base import expand_dims
import os
import os.path as osp

import algorithm
from tools.experiment_grid import ExperimentGrid

SUBSTITUTIONS = {
    'env': 'env_name',
    'hid': 'ac_kwargs:hidden_sizes',
    'act': 'ac_kwargs:activation',
    'cpu': 'num_cpu',
    'dt': 'datestamp'
}
RUN_KEYS = ['num_cpu', 'data_dir', 'datestamp']

# Only some algorithms can be parallelized (have num_cpu > 1):
MPI_COMPATIBLE_ALGOS = ['vpg', 'trpo', 'ppo']


def add_with_backends(algo_list):
    # helper function to build lists with backend-specific function names
    algo_list_with_backends = deepcopy(algo_list)
    for algo in algo_list:
        algo_list_with_backends += [algo + '_tf1', algo + '_pytorch']
    return algo_list_with_backends


def friendly_error(err_msg):
    return "\n\n" + err_msg + "\n\n"


def process_args(exp, args):
    def process(arg):
        """
        使用eval函数处理参数, 使得传参可以传递函数.
        """
        try:
            return eval(arg)
        except:
            return arg

    arg_dict = dict()
    arg_key = ""
    for i, arg in enumerate(args):
        assert i > 0 or '--' in arg, friendly_error("You didn't specify a first flag.")
        if '--' in arg:
            arg_key = arg.lstrip('-')
            arg_dict[arg_key] = []
        else:
            arg_dict[arg_key].append(process(arg))

    for k, v in arg_dict.items():
        if len(v) == 0:
            v.append(True)

    given_shorhands = dict()
    fixed_keys = list(arg_dict.keys())
    for k in fixed_keys:
        p1, p2 = k.find('['), k.find(']')
        if p1 >= 0 and p2 >= 0:
            k_new = k[:p1]
            shorthand = k[p1 + 1:p2]
            given_shorhands[k_new] = shorthand
            arg_dict[k_new] = arg_dict[k]
            del arg_dict[k]
    # 允许简称传递参数, 比如"env"代表"env_name"
    for special_name, true_name in SUBSTITUTIONS.items():
        if special_name in arg_dict:
            arg_dict[true_name] = arg_dict[special_name]
            del arg_dict[special_name]
        if special_name in given_shorhands:
            given_shorhands[true_name] = given_shorhands[special_name]
            del given_shorhands[special_name]
    # 为了进行网格调参,把参数离散化
    run_kwargs = dict()
    for k in RUN_KEYS:
        if k in arg_dict:
            val = arg_dict[k]
            assert len(val) == 1, friendly_error("You can only provide one value for %s." % k)
            run_kwargs[k] = val[0]
            del arg_dict[k]
    # 确定实验名称
    if 'exp_name' in arg_dict:
        assert len(arg_dict['exp_name']) == 1, friendly_error("只能给定一个名字.")
        exp_name = arg_dict['exp_name'][0]
        del arg_dict['exp_name']
    else:
        exp_name = "cmd_" + exp
    # 并行执行参数, ddpg算法不需要,为了以后复制方便这里写上
    if "num_cpu" in run_kwargs and not (run_kwargs['num_cpu'] == 1):
        assert exp in MPI_COMPATIBLE_ALGOS, friendly_error("该算法不支持并行训练")

    valid_envs = [e.id for e in list(gym.envs.registry.all())]
    assert "env_name" in arg_dict, friendly_error("You did not give a value for --env_name!")
    for env_name in arg_dict['env_name']:
        err_msg = dedent("""

            %s is not registered with Gym.

            Recommendations:

                * Check for a typo (did you include the version tag?)

                * View the complete list of valid Gym environments at

                    https://gym.openai.com/envs/

            """ % env_name)
        assert env_name in valid_envs, err_msg
    return exp_name, arg_dict, run_kwargs, given_shorhands


def run_grid_search(exp, args):
    """
    网格调参
    """
    alg = "algorithm.%s" % exp
    run_alg = eval(alg)

    exp_name, arg_dict, run_kwargs, given_shorhands = process_args(exp, args)
    # print(run_kwargs, given_shorhands)
    eg = ExperimentGrid(name=exp_name)
    for k, v in arg_dict.items():
        eg.add(k, v, shorthand=given_shorhands.get(k))
    # print(eg.variants())
    eg.run(run_alg, **run_kwargs)


if __name__ == "__main__":
    """
    python -m spinup.run ppo --exp_name ppo_ant --env Ant-v2 --clip_ratio 0.1 0.2
        --hid[h] [32,32] [64,32] --act torch.nn.Tanh --seed 0 10 20 --dt
        --data_dir path/to/data
    """
    run_type = sys.argv[2]
    exp = sys.argv[4]
    if run_type == "train":
        # 网格搜索超参训练
        # CarRacing-v0
        # LunarLander-v2
        cmd = """
            python run.py --exp_name %s_Pendulum1 --env Pendulum-v0
                --seed 0 --data_dir data --dt --num_cpu 6
            """ % (exp)
        args = cmd.strip().split()[2:]
        run_grid_search(exp, args)
    elif run_type == "plot":
        runfile = osp.join(osp.abspath(osp.dirname(__file__)), "tools", "plot.py")
        cmd = """
            python plot.py --savedir=results
            data/2021-01-20_vpg_Pendulum1
            """
        cmd = cmd.strip().split()
        args = [sys.executable if sys.executable else "python", runfile] + cmd[2:]
        subprocess.check_call(args, env=os.environ)
    elif run_type == "test":
        runfile = osp.join(osp.abspath(osp.dirname(__file__)), "test_policy.py")
        cmd = """
            python test_policy.py /home/user/pro/mygithub/learn2spinningup/data/2021-01-20_vpg_Pendulum1/2021-01-20_22-36-03-vpg_Pendulum1_s0
            --norender --episodes=100 --len=1000
            """
        cmd = cmd.strip().split()
        args = [sys.executable if sys.executable else "python", runfile] + cmd[2:]
        subprocess.check_call(args, env=os.environ)
    else:
        friendly_error("参数错误!")
        exit()
