import numpy as np
import torch
from torch.optim import Adam
import gym
from copy import deepcopy
import time

from tools.mpi_utils import mpi_statistics_scalar, mpi_fork, mpi_avg, proc_id, num_procs
from tools.utils import setup_logger_kwargs, EpochLogger, combined_shape
from tools.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from tools.config import devices
from model import count_vars, discount_cumsum, MLPActorCritic
