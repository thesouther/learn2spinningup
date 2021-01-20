import multiprocessing
import numpy as np
import os
import torch
from mpi4py import MPI
from .mpi_utils import broadcast, mpi_avg, num_procs, proc_id


def setup_pytorch_for_mpi():
    """
    避免每个torch进行使用超限的CPU资源导致降速
    """
    print('Proc %d: Reporting original number of Torch threads as %d.' % (proc_id(), torch.get_num_threads()),
          flush=True)
    if torch.get_num_threads() == 1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / num_procs()), 1)
    torch.set_num_threads(fair_num_threads)
    print('Proc %d: Reporting original number of Torch threads as %d.' % (proc_id(), torch.get_num_threads()),
          flush=True)


def mpi_avg_grads(module):
    """
    把梯度对线程数量做平均
    """
    if num_procs() == 1:
        return
    for p in module.parameters():
        p_grad_numpy = p.grad.numpy()
        avg_p_grad = mpi_avg(p.grad)
        p_grad_numpy[:] = avg_p_grad[:]


def sync_params(module):
    """
    同步所有线程的参数
    """
    if num_procs() == 1:
        return
    for p in module.parameters():
        p_numpy = p.data.numpy()
        broadcast(p_numpy)