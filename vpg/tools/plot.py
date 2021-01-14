from posix import listdir
from pandas.io import parsers
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np
import time

DIV_LINE_WIDTH = 50

# 载入数据时的辅助信息
exp_idx = 0
units = dict()


def plot_data(data, xaxis="Epoch", value="AverageEpRet", condition="Condition1", smooth=1, **kwargs):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set(style="darkgrid", font_scale=1.5)
    sns.tsplot(data=data, time=xaxis, value=value, unit="Unit", condition=condition, ci="sd", **kwargs)
    """
    如果使用的seaborn版本大于0.8.1, 改成
        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)
    改一下colorscheme and the default legend style
    """
    plt.legend(loc="best").set_draggable(True)
    #plt.legend(loc='upper center', ncol=3, handlelength=1,
    #           borderaxespad=0., prop={'size': 13})
    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    swap L38 with:

    plt.legend(loc='upper center', ncol=6, handlelength=1,
            mode="expand", borderaxespad=0., prop={'size': 13})
    """

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.tight_layout(pad=0.5)


def get_datasets(logdir, condition=None):
    """
    导入Logger生成的logdir文件
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if "progress.txt" in files:
            exp_name = None
            try:
                config_path = open(osp.join(root, "config.json"))
                config = json.load(config_path)
                if "exp_name" in config:
                    exp_name = config["exp_name"]
            except:
                print("NO file named config.json")
            condition1 = condition or exp_name or "exp"
            condition2 = condition1 + "-" + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            try:
                exp_data = pd.read_table(osp.join(root, "progress.txt"))
            except:
                print("Could not read from %s" % osp.join(root, "progress.txt"))
                continue
            performance = "AverageTestEpRet" if "AverageTestEpRet" in exp_data else "AverageEpRet"
            exp_data.insert(len(exp_data.columns), "Unit", unit)
            exp_data.insert(len(exp_data.columns), "Condition1", condition1)
            exp_data.insert(len(exp_data.columns), "Condition2", condition2)
            exp_data.insert(len(exp_data.columns), "Performance", exp_data[performance])
            datasets.append(exp_data)
    return datasets


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is, 
           pull data from it; 

        2) if not, check to see if the entry is a prefix for a 
           real directory, and pull data from that.
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1] == os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x: osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir = os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])
    """
    强制执行选择规则，该规则检查logdir中是否包含某些子串。 
    如果您一次启动多个具有相似名称的job，则可以轻松查看特定消融实验的图形。
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not (x in log) for x in exclude)]

    print('Plotting from...\n' + '=' * DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print("\n" + "=" * DIV_LINE_WIDTH)
    # Make sure the legend is compatible with the logdirs
    assert not (legend) or (len(legend) == len(logdirs)), "必须每个实验一个图例"
    # load data
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg)
    else:
        for log in logdirs:
            data += get_datasets(log)
    return data


def make_plots(all_logdirs,
               savedir="results/",
               legend=None,
               xaxis=None,
               values=None,
               count=False,
               font_scale=1.5,
               smooth=1,
               select=None,
               exclude=None,
               estimator="mean"):
    data = get_all_datasets(all_logdirs, legend, select, exclude)
    values = values if isinstance(values, list) else [values]
    conditon = "Condition2" if count else "Condition1"
    # choose what to show on main curve: mean? max? min?
    estimator = getattr(np, estimator)
    for value in values:
        plt.figure()
        plot_data(data, xaxis=xaxis, value=value, condition=conditon, smooth=smooth, estimator=estimator)

    exp_name = list(filter(lambda x: x != "", all_logdirs[0].split(os.sep)))[-1]
    ymd_time = time.strftime("%Y-%m-%d_")
    plt_save_path = osp.join(savedir, ymd_time + exp_name + ".png")
    plt.savefig(plt_save_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", nargs='*')
    parser.add_argument("--savedir", default="results")
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    parser.add_argument('--value', '-y', default='Performance', nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()
    make_plots(args.logdir,
               args.savedir,
               args.legend,
               args.xaxis,
               args.value,
               args.count,
               smooth=args.smooth,
               select=args.select,
               exclude=args.exclude,
               estimator=args.est)


if __name__ == "__main__":
    main()