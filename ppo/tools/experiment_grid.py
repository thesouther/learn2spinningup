import numpy as np
import os, sys
import os.path as osp
import string
from textwrap import dedent
from numpy.lib.arraysetops import isin
from tqdm import trange
import time
import psutil
import json
import subprocess
from subprocess import CalledProcessError
import cloudpickle
import base64
import zlib
from .utils import colorize, setup_logger_kwargs
from .json_utils import convert_json
from .config import DEFAULT_SHORTHAND, WAIT_BEFORE_LAUNCH
from .mpi_utils import mpi_fork

DIV_LINE_WIDTH = 80


def call_experiment(
        exp_name,
        thunk,
        seed=0,
        num_cpu=1,
        data_dir=None,
        datestamp=False,
        # use_gpu=False,
        **kwargs):
    """
    使用超参数和configuration运行函数
    """
    num_cpu = psutil.cpu_count(logical=False) if num_cpu == "auto" else num_cpu
    kwargs["seed"] = seed
    # kwargs["use_gpu"] = use_gpu
    print(colorize("Running experiment:\n", color="cyan", bold=True))
    print(exp_name + "\n")
    print(colorize("with kwargs:\n", color="cyan", bold=True))
    kwargs_json = convert_json(kwargs)
    print(json.dumps(kwargs_json, separators=(",", ":\t"), indent=4, sort_keys=True))
    print("\n")

    if "logger_kwargs" not in kwargs:
        kwargs["logger_kwargs"] = setup_logger_kwargs(exp_name, seed, data_dir, datestamp)
    else:
        print("Note: Call experiment is not handling logger_kwargs.\n")

    def thunk_plus():
        if "env_name" in kwargs:
            import gym
            env_name = kwargs["env_name"]
            kwargs["env_fn"] = lambda: gym.make(env_name)
            del kwargs["env_name"]
        # Fork into multiple processes
        mpi_fork(num_cpu)
        thunk(**kwargs)

    pickled_thunk = cloudpickle.dumps(thunk_plus)
    encoded_thunk = base64.b64encode(zlib.compress(pickled_thunk)).decode("utf-8")
    # 获取上一层目录下的 run_entrypoint.py 文件
    entrypoint = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))), "run_entrypoint.py")
    cmd = [sys.executable if sys.executable else 'python', entrypoint, encoded_thunk]
    try:
        subprocess.check_call(cmd, env=os.environ)
    except CalledProcessError:
        err_msg = "\n" * 3 + "=" * DIV_LINE_WIDTH + "\n" + dedent("""
            There appears to have been an error in your experiment.

            Check the traceback above to see what actually went wrong. The 
            traceback below, included for completeness (but probably not useful
            for diagnosing the error), shows the stack leading up to the 
            experiment launch.

        """) + "=" * DIV_LINE_WIDTH + "\n" * 3
        print(err_msg)
        raise

    logger_kwargs = kwargs["logger_kwargs"]
    plot_cmd = "python plot.py " + logger_kwargs["output_dir"]
    plot_cmd = colorize(plot_cmd, "green")
    test_cmd = "python test_policy.py " + logger_kwargs["output_dir"]
    test_cmd = colorize(test_cmd, "green")
    output_msg = "\n" * 5 + "=" * DIV_LINE_WIDTH + "\n" + dedent("""\
    End of experiment.


    Plot results from this run with:

    %s


    Watch the trained agent with:

    %s


    """ % (plot_cmd, test_cmd)) + "=" * DIV_LINE_WIDTH + "\n" * 5
    print(output_msg)


def all_bools(vals):
    return all([isinstance(v, bool) for v in vals])


def valid_str(v):
    """
    将一个或多个值转换为可以放入文件路径的字符串
    """
    if hasattr(v, "__name__"):
        return valid_str(v.__name__)
    if isinstance(v, tuple) or isinstance(v, list):
        return "-".join([valid_str(x) for x in v])
    str_v = str(v).lower()
    valid_chars = "-_%s%s" % (string.ascii_letters, string.digits)
    str_v = "".join(c if c in valid_chars else "-" for c in str_v)
    return str_v


class ExperimentGrid:
    """
    使用grid search 对超参数调优
    """
    def __init__(self, name=''):
        self.keys = []
        self.vals = []
        self.shs = []
        self.in_names = []
        self.name(name)

    def name(self, _name):
        assert isinstance(_name, str), "Name has to be a string."
        self._name = _name

    def print(self):
        """
        输出相关调参信息
        """
        print("=" * DIV_LINE_WIDTH)
        # 如果实验名字短,打印一行,如果太长打印两行
        base_msg = "ExperimentGrid %s runs over parameters: \n"
        name_insert = "[" + self._name + "]"
        if len(base_msg % name_insert) <= 80:
            msg = base_msg % name_insert
        else:
            msg = base_msg % (name_insert + "\n")
        print(colorize(msg, color="green", bold=True))

        # list off parameters, shorthands, and possible values.
        for k, v, sh in zip(self.keys, self.vals, self.shs):
            color_k = colorize(k.ljust(40), color="cyan", bold=True)
            print("", color_k, "[" + sh + "]" if sh is not None else "", "\n")
            for i, val in enumerate(v):
                print("\t" + str(convert_json(val)))
            print()

        # 计算变量数量
        nvars_total = int(np.prod([len(v) for v in self.vals]))
        if "seed" in self.keys:
            num_seeds = len(self.vals[self.keys.index("seed")])
            nvars_seedless = int(nvars_total / num_seeds)
        else:
            nvars_seedless = nvars_total
        print(" Variants, counting seeds: ".ljust(40), nvars_total)
        print(" Variants, not counting seeds: ".ljust(40), nvars_seedless)
        print()
        print("=" * DIV_LINE_WIDTH)

    def _default_shorthand(self, key):
        """
        对key做一个shorthand, 用冒号分隔后的每一部分的前三个字母表示, 
        如果前三个字符包含非数字字母, 就直接截掉
        """
        valid_chars = "%s%s" % (string.ascii_letters, string.digits)

        def shear(x):
            return "".join(z for z in x[:3] if z in valid_chars)

        sh = "-".join([shear(x) for x in key.split(":")])
        return sh

    def add(self, key, vals, shorthand=None, in_name=False):
        """
        Add a parameter (key) to the grid config, with potential values (vals).
        Args:
            key (string): Name of parameter.

            vals (value or list of values): Allowed values of parameter.

            shorthand (string): Optional, shortened name of parameter. For 
                example, maybe the parameter ``steps_per_epoch`` is shortened
                to ``steps``. 

            in_name (bool): When constructing variant names, force the
                inclusion of this parameter into the name.
        """
        assert isinstance(key, str), "Key must be a string."
        assert shorthand is None or isinstance(shorthand, str), "Shorthand must be a string."
        if not isinstance(vals, list):
            vals = [vals]
        if DEFAULT_SHORTHAND and shorthand is None:
            shorthand = self._default_shorthand(key)
        self.keys.append(key)
        self.vals.append(vals)
        self.shs.append(shorthand)
        self.in_names.append(in_name)

    def variant_name(self, variant):
        """
        给定一个variant变量(有效参数/值组成的字典)，创建exp_name
        用variant构造grid_name, 形式为[variant]_[para_name/shorthand]_[values]
        """
        def get_val(v, k):
            """
            从给定变量字典中获得正确值的实用方法.
            假设参数k描述到嵌套字典的路径，例如k='a:b:c'对应于value=变量['a']['b']['c'].
            """
            if k in v:
                return v[k]
            else:
                splits = k.split(":")
                k0, k1 = splits[0], ":".join(splits[1:])
                return get_val(v[k0], k1)

        var_name = self._name
        for k, v, sh, inn in zip(self.keys, self.vals, self.shs, self.in_names):
            if (len(v) > 1 or inn) and not (k == "seed"):
                # 使用shorthand, 如果不行就用全名
                param_name = sh if sh is not None else k
                param_name = valid_str(param_name)
                # 对参数k,获取其变量值
                variant_val = get_val(variant, k)
                if all_bools(v):
                    var_name += ("_" + param_name) if variant_val else ""
                else:
                    var_name += "_" + param_name + valid_str(variant_val)

        return var_name.lstrip("_")

    def _variants(self, keys, vals):
        """
        递归地构建有效变量列表
        """
        if len(keys) == 1:
            pre_variants = [dict()]
        else:
            pre_variants = self._variants(keys[1:], vals[1:])
        variants = []
        for val in vals[0]:
            for pre_v in pre_variants:
                v = {}
                v[keys[0]] = val
                v.update(pre_v)
                variants.append(v)
        return variants

    def variants(self):
        """
        构造一个字典列表, 每个字典是grid里一个有效的config.
        对于名称为 ``'full:param:name'``形式的变量参数有特殊的处理.
        冒号表示参数有内嵌字典结构, 例如
            ====================  ===
            Key                   Val
            ====================  ===
            ``'base:param:a'``    1
            ``'base:param:b'``    2
            ====================  ===
        表明变量字典的结构为
            variant = {
                base: {
                    param : {
                        a : 1,
                        b : 2
                        }
                    }    
                }

        """
        flat_variants = self._variants(self.keys, self.vals)

        def unflatten_var(var):
            """
            基于key的name, 构建完整结构的字典
            """
            new_var = dict()
            unflatten_set = set()
            for k, v in var.items():
                if ":" in k:
                    splits = k.split(":")
                    k0 = splits[0]
                    assert k0 not in new_var or isinstance(new_var[k0], dict), \
                        "同一个key不能设置多个值."
                    if not (k0 in new_var):
                        new_var[k0] = dict()
                    sub_k = ":".join(splits[1:])
                    new_var[k0][sub_k] = v
                    unflatten_set.add(k0)
                else:
                    assert not (k in new_var), "同一个key不能设置多个值."
                    new_var[k] = v
            for k in unflatten_set:
                new_var[k] = unflatten_var(new_var[k])
            return new_var

        new_variants = [unflatten_var(var) for var in flat_variants]
        return new_variants

    def run(
        self,
        thunk,
        num_cpu=1,
        data_dir=None,
        datestamp=False,
        # use_gpu=False
    ):
        """
        使用'thunk'函数运行grid中每一个变量.
        'thunk' 必须是可调用函数或者字符串, 如果是string, 必须是参数的名字, 其值都是可调用函数.
        使用``call_experiment`` 运行实验, 并使用``self.variant_name()``给定实验名字.

        注意:ExperimentGrid.run与call_experiment的参数相同,`seed`除外
        """
        self.print()
        # list 所有变量
        variants = self.variants()
        #输出变量名
        var_names = set([self.variant_name(var) for var in variants])
        var_names = sorted(list(var_names))
        line = "=" * DIV_LINE_WIDTH
        preparing = colorize("Preparing to run the following experiment...", color="green", bold=True)
        joined_var_names = '\n'.join(var_names)
        announcement = f"\n{preparing}\n\n{joined_var_names}\n\n{line}"
        print(announcement)

        if WAIT_BEFORE_LAUNCH > 0:
            delay_msg = colorize(dedent("""
            Launch delayed to give you a few seconds to review your experiments.

            To customize or disable this behavior, change WAIT_BEFORE_LAUNCH in
            spinup/config.py.d

            """),
                                 color="cyan",
                                 bold=True) + line
            print(delay_msg)
            wait, steps = WAIT_BEFORE_LAUNCH, 100
            prog_bar = trange(steps,
                              desc="launching in...",
                              leave=False,
                              ncols=DIV_LINE_WIDTH,
                              mininterval=0.25,
                              bar_format="{desc}: {bar}| {remaining} {elapsed}")
            for _ in prog_bar:
                time.sleep(wait / steps)
        for var in variants:
            exp_name = self.variant_name(var)
            if isinstance(thunk, str):
                thunk_ = var[thunk]
                del var[thunk]
            else:
                thunk_ = thunk

            call_experiment(
                exp_name=exp_name,
                thunk=thunk_,
                num_cpu=num_cpu,
                data_dir=data_dir,
                datestamp=datestamp,
                # use_gpu=use_gpu,
                **var)


def test_eg():
    eg = ExperimentGrid()
    eg.add("test:a", [1, 2, 3], "ta", True)
    eg.add("test:b", [1, 2, 3])
    eg.add("some", [4, 5])
    eg.add("why", [True, False])
    eg.add("huh", 5)
    eg.add("no", 6, in_name=True)
    return eg.variants()


if __name__ == "__main__":
    a = test_eg()
    print(a)
    b = [{
        'clip_ratio': 0.1,
        'seed': 0,
        'env_name': 'Ant-v2',
        'ac_kwargs': {
            'hidden_sizes': [32, 32],
            'activation': ""
        }
    }, {
        'clip_ratio': 0.1,
        'seed': 0,
        'env_name': 'Ant-v2',
        'ac_kwargs': {
            'hidden_sizes': [64, 32],
            'activation': ""
        }
    }, {
        'clip_ratio': 0.1,
        'seed': 10,
        'env_name': 'Ant-v2',
        'ac_kwargs': {
            'hidden_sizes': [32, 32],
            'activation': ""
        }
    }, {
        'clip_ratio': 0.1,
        'seed': 10,
        'env_name': 'Ant-v2',
        'ac_kwargs': {
            'hidden_sizes': [64, 32],
            'activation': ""
        }
    }, {
        'clip_ratio': 0.1,
        'seed': 20,
        'env_name': 'Ant-v2',
        'ac_kwargs': {
            'hidden_sizes': [32, 32],
            'activation': ""
        }
    }, {
        'clip_ratio': 0.1,
        'seed': 20,
        'env_name': 'Ant-v2',
        'ac_kwargs': {
            'hidden_sizes': [64, 32],
            'activation': ""
        }
    }, {
        'clip_ratio': 0.2,
        'seed': 0,
        'env_name': 'Ant-v2',
        'ac_kwargs': {
            'hidden_sizes': [32, 32],
            'activation': ""
        }
    }, {
        'clip_ratio': 0.2,
        'seed': 0,
        'env_name': 'Ant-v2',
        'ac_kwargs': {
            'hidden_sizes': [64, 32],
            'activation': ""
        }
    }, {
        'clip_ratio': 0.2,
        'seed': 10,
        'env_name': 'Ant-v2',
        'ac_kwargs': {
            'hidden_sizes': [32, 32],
            'activation': ""
        }
    }, {
        'clip_ratio': 0.2,
        'seed': 10,
        'env_name': 'Ant-v2',
        'ac_kwargs': {
            'hidden_sizes': [64, 32],
            'activation': ""
        }
    }, {
        'clip_ratio': 0.2,
        'seed': 20,
        'env_name': 'Ant-v2',
        'ac_kwargs': {
            'hidden_sizes': [32, 32],
            'activation': ""
        }
    }, {
        'clip_ratio': 0.2,
        'seed': 20,
        'env_name': 'Ant-v2',
        'ac_kwargs': {
            'hidden_sizes': [64, 32],
            'activation': ""
        }
    }]
    print(len(b))
