# -*- coding: utf-8 -*-

from sklearn.svm import SVR 
import h5py
import pickle
import numpy as np
from pandas import to_datetime
import pandas as pd
from sklearn import cluster
import holidays



class MinMaxNorm01(object):
    def __init__(self):
        pass

    def fit(self, x):
        self.min = x.min()
        self.max = x.max()
        #print('Min:{}, Max:{}'.format(self.min, self.max))

    def transform(self, x):
        x = 1.0 * (x - self.min) / (self.max - self.min)
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        x = x * (self.max - self.min) + self.min
        return x
class MinMaxNorm02(object):
    def __init__(self):
        pass

    def fit(self, x):
        self.min = x.min()
        self.max = x.max()
        #print('Min:{}, Max:{}'.format(self.min, self.max))

    def transform(self, x):
        x = 1.0 * (x - self.min) / (self.max - self.min)
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        x = x * (self.max - self.min) + self.min
        return x

def get_date_feature(idx):
    # 创建意大利节假日对象
    it_holidays = holidays.Italy(years=[2013, 2014])
    weekday = idx.weekday()
    hour = idx.hour
    is_saturday = idx.weekday() // 6
    is_sunday = idx.weekday() // 7
    is_holiday = int(idx in it_holidays) * 5
    return weekday, hour, is_saturday, is_sunday, is_holiday


def traffic_loader(f, feature_path, opt):
    feature_names = ['social', 'BSs', 'poi_1', 'poi_2']
    feature_data = pd.read_csv(feature_path, header=0)
    feature_data.columns = feature_names

    feature = np.reshape(feature_data.values, (opt.height, opt.width, 4))
    # print(f['data'].shape)
    if opt.nb_flow == 1:
        if opt.traffic == 'sms':
            # data = f['data'][7*24:, :, 0]
            # data_p = f['data'][6*24:-1*24, :, 0]
            # data_t = f['data'][:-7*24, :, 0]
            data = f['data'][:, :, 0]/2.0#+f['data'][:, :, 1]
        elif opt.traffic == 'call':
            data = f['data'][:, :, 1]/2.0#+f['data'][:, :, 3]
        elif opt.traffic == 'internet':
            data = f['data'][:, :, 2]/2.0
        else:
            raise IOError("Unknown traffic type")
        result = data.reshape((-1, 1, opt.height, opt.width))
        # result_o = data.reshape((-1, 1, opt.height, opt.width))
        # # print(result_o.shape)
        # result_p = data_p.reshape((-1, 1, opt.height, opt.width))
        # result_t = data_t.reshape((-1, 1, opt.height, opt.width))
        # result = np.concatenate((result_o, result_p, result_t), axis=1)

        if opt.crop:
            result = result[:, :, opt.rows[0]:opt.rows[1], opt.cols[0]:opt.cols[1]]
            feature = feature[opt.rows[0]:opt.rows[1], opt.cols[0]:opt.cols[1], :]
        return result, feature

    elif opt.nb_flow == 2:
        if opt.traffic == 'sms':
            data = f['data'][:,:,:2]
            result = data.reshape((-1, 2, opt.height, opt.width))
        elif opt.traffic == 'call':
            data = f['data'][:,:,2:4]
            result = data.reshape((-1, 2, opt.height, opt.width))
        elif opt.traffic == 'internet':
            data = f['data'][:, :, 4]
            result = data.reshape((-1, 1, opt.height, opt.width))
        else:
            raise IOError("Unknown traffic type")
        
        if opt.crop:
            result = result[:, :, opt.rows[0]:opt.rows[1], opt.cols[0]:opt.cols[1]]
            feature = feature[opt.rows[0]:opt.rows[1], opt.cols[0]:opt.cols[1], :]
        return result, feature

    else:
        print("Wrong parameter with nb_flow")
        exit(0)


def get_label(data, feature, index, clusters):
    samples, channels, h, w = data.shape
    sum_data = np.sum(data, axis=1).reshape((samples, h*w))
    df_data = pd.DataFrame(sum_data, index=index)

    df_data = df_data.resample('1D').sum().transpose()

    feature = pd.DataFrame(np.reshape(feature, (h*w, -1)))
    df = pd.concat([df_data, feature], axis=1)
    df.fillna(0, inplace=True)
    # clustering the data points and get the cluster id
    clf = cluster.AgglomerativeClustering(n_clusters=clusters)
    clf.fit(df)
    return clf.labels_

def get_label_v2(data, feature, index, clusters):
    # print(np.sum(feature, axis=-1))
    # print(data.sum().sum())
    all_data = data.sum().sum() + np.sum(feature, axis=-1) #(20,20)
    mask = all_data.astype(bool)
    # print(mask)
    from sklearn.feature_extraction import image
    graph = image.img_to_graph(all_data, mask=mask)
    # print(graph)
    graph.data = np.exp(-graph.data / graph.data.std())
    # print(graph)
    labels = cluster.spectral_clustering(graph, n_clusters=clusters, eigen_solver='arpack')
    # print(labels)
    return labels

def read_data(path, feature_path, opt):
    f = h5py.File(path, 'r')
    data, feature_data = traffic_loader(f, feature_path, opt)
    print('data shape:', data.shape)
    # index = f['idx'].value.astype(str)
    index = f['idx'][:].astype(str)
    index = to_datetime(index, format='%Y-%m-%d %H:%M')
    # 打印一次index的值
    flag = True
    if flag == False:
        print(index)
        flag = True
    cell_label = get_label_v2(data, feature_data, index, opt.cluster)
    mmn = MinMaxNorm01()
    data_scaled = mmn.fit_transform(data)
    
    print('data_scaled shape:', data_scaled.shape)
    X, y = [], []
    X_meta = []

    h, w = data.shape[2], data.shape[3]
    print('len(data):', len(data))
    for i in range(opt.close_size, len(data)):
        xc_ = [data_scaled[i - c][:,:,:] for c in range(1, opt.close_size + 1)]
        # xc_.append(data_scaled[i][1:2,:,:])
        # xc_.append(data_scaled[i][2:3,:,:])
        xc_ = np.asarray(xc_)
        a, b, c, d, e = get_date_feature(index[i])
        # c, d, e = get_date_feature(index[i])
        X_meta.append((a, b, c, d, e))
        # X_meta.append((c, d, e))

        if opt.close_size > 0:
            X.append(xc_)

        y.append(data_scaled[i][:,:,:])
    print('xc_ shape:', xc_.shape)
    X = np.asarray(X)
    print("X.shape",X.shape)
    X_meta = np.asarray(X_meta)
    mmn2 = MinMaxNorm02()
    X_meta = mmn2.fit_transform(X_meta)
    X_cross = np.asarray(feature_data)
    X_cross = np.reshape(X_cross, (h * w, -1))
    y = np.asarray(y)
    print('Y shape:', y.shape)

    X_cross = np.moveaxis(X_cross, 0, -1)
    X_crossdata = np.repeat(X_cross, X.shape[0]).reshape((-1, 4, h, w))

    #print('X shape:' + str(X.shape))
    #print('X meta shape:' + str(X_meta.shape))
    #print('X cross shape:' + str(X_crossdata.shape))

    return X, X_meta, X_crossdata, y, cell_label, mmn