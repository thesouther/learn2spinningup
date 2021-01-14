import os, sys
import os.path as osp
import torch
# 数据和log文件地址
DEFAULT_DATA_DIR = osp.join('data')
# 是否在log文件名后加时间戳
FORCE_DATESTAMP = True
#使用网格法调参时， 是否使用shorthand
DEFAULT_SHORTHAND = True
# 使用网格调参时，等待多长时间再启动实验
WAIT_BEFORE_LAUNCH = 5

devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
