import os
os.environ['OMP_NUM_THREADS'] = '2'
import sys
import argparse
import numpy as np
from datetime import datetime
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# sys.path.append('../../')
# sys.path.append('G:\Code\study\GNN\MVSTGN')
from utils.dataset import read_data
from utils.model import DenseNet
# from utils.spatial_lstm import Mvstgn
from utils.Mvstgn import Mvstgn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import h5py
import time
from utils.print_para import print_para
from sklearn.svm import SVR 
import h5py
import pickle
import numpy as np
from pandas import to_datetime
import pandas as pd
from sklearn import cluster
import holidays
it_holidays = holidays.Italy(years=[2013, 2014])
print(it_holidays)
f = h5py.File('dataset\data_git_version.h5', 'r')
def get_date_feature(idx):
    # 创建意大利节假日对象
    it_holidays = holidays.Italy(years=[2013, 2014])
    a = idx.weekday()
    b = idx.hour
    c = idx.weekday() // 6 * 3
    d = idx.weekday() // 7 * 3
    e = int(idx in it_holidays) * 10
    return a, b, c, d, e
index = f['idx'][:].astype(str)
index = to_datetime(index, format='%Y-%m-%d %H:%M')
print(index)
X_meta = []
for i in range(3, len(index)):
    # xc_ = [data_scaled[i - c][:,:,:] for c in range(1, opt.close_size + 1)]
    # xc_.append(data_scaled[i][1:2,:,:])
    # xc_.append(data_scaled[i][2:3,:,:])
    # xc_ = np.asarray(xc_)
    a, b, c, d, e = get_date_feature(index[i])
    X_meta.append((a, b, c, d, e))

# print(X_meta[0:100])

""" data = f['data'][:].astype(np.float32)
type = 'sms'
if type == 'sms':
    data = data[:,:,0]
    result = data.reshape((-1, 1, 100, 100))
print(result.shape)
# 提取一个节点的数据
node = 0
node_data = result[:, :, node, node]
print(node_data.shape)
# 画图
# x axis values is index
# corresponding y axis values is node_data 
plt.plot(index, node_data)
plt.show() """
