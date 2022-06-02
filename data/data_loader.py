
import os
import warnings

from datetime import timedelta

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils.tools import StandardScaler

warnings.filterwarnings('ignore')

class Dataset_ETT(Dataset):
    def __init__(self, root_path,  flag='train', features=None, size=None,
                 data_path=['china.csv', 'ETTm1.csv', 'ETTm2.csv','ETTh1.csv','ETTh2.csv'],
                 target=None,scale=True, inverse=False,granularity=96, data_index=1,**kwargs):
        # size [pre_len, lookback_len, horizon_len, post_len]
        # info
        if size == None:
            self.pre_len = 31
            self.lookback_len = 7
            self.horizon_len = 1
            self.post_len = 31
        else:
            self.pre_len = size[0]
            self.lookback_len = size[1]
            self.horizon_len = size[2]
            self.post_len = size[3]
        self.total_len = self.pre_len + self.lookback_len + self.horizon_len + self.post_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        assert granularity == 24 or granularity == 96
        self.granularity = granularity
        assert data_index in [1,2,3,4]
        self.data_index = data_index
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse

        self.root_path = root_path
        self.data_path = data_path

        self.__load_data__()

    def __load_data__(self):
        self.scaler = StandardScaler()
        self.data_x = pd.read_csv(os.path.join(os.path.split(self.root_path)[0],
                                               self.data_path[0]), index_col='date', parse_dates=True,
                                  usecols=self.features)

        if self.granularity == 96:
            cut = 80
        else:cut = 20

        if self.target:
            self.target = ['date', self.target]

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path[self.data_index]),usecols=self.target, parse_dates=[0], index_col=[0])[:-cut]
        df_raw = df_raw.to_numpy().reshape((-1, self.granularity, df_raw.shape[-1]))

        border1s = [0, 12 * 30 - self.lookback_len, 12 * 30 + 4 * 30 - self.lookback_len]
        border2s = [12 * 30, 12 * 30 + 4 * 30, 12 * 30 + 8 * 30]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        data = df_raw[:, :, :]

        self.data_stamp = pd.date_range('2016-07-01', '2018-06-25', freq='D').to_series()[border1:border2].reset_index(
            drop=True)

        if self.scale:
            scale_data = data[:border2s[0], :, :]
            self.scaler.fit(scale_data.reshape((-1, data.shape[-1])))
            data = self.scaler.transform(data)

        if self.inverse:
            self.data_y = df_raw[border1:border2]
        else:
            self.data_y = data[border1:border2]

    def __getitem__(self, index):

        date = self.data_stamp[index].date()
        start = date - timedelta(days=self.pre_len)
        end = date + timedelta(days=self.lookback_len+self.horizon_len+self.post_len-1)
        x_date = self.data_x[start:end].values
        assert len(x_date) == self.total_len
        x_lookback = self.data_y[index:index+self.lookback_len]
        y_horizon = self.data_y[index+self.lookback_len:index + self.lookback_len + self.horizon_len]

        return x_date, x_lookback, y_horizon

    def __len__(self):
        return len(self.data_y)-self.lookback_len-self.horizon_len+1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ECL(Dataset):
    def __init__(self, root_path, flag='train', features=None, size=None,
                 data_path=['portugal.csv', 'electricity.csv'],target=None,scale=True, inverse=False,**kwargs):
        # size [pre_len, lookback_len, horizon_len, post_len]
        # info
        if size == None:
            self.pre_len = 31
            self.lookback_len = 7
            self.horizon_len = 1
            self.post_len = 31
        else:
            self.pre_len = size[0]
            self.lookback_len = size[1]
            self.horizon_len = size[2]
            self.post_len = size[3]
        self.total_len = self.pre_len + self.lookback_len + self.horizon_len + self.post_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target

        self.scale = scale
        self.inverse = inverse

        self.root_path = root_path
        self.data_path = data_path

        self.__load_data__()

    def __load_data__(self):

        self.scaler = StandardScaler()
        self.data_x = pd.read_csv(os.path.join(os.path.split(self.root_path)[0],
                                               self.data_path[0]), index_col='date', parse_dates=True,
                                  usecols=self.features)

        if self.target:
            self.target = ['date', self.target]

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path[1]),usecols=self.target,index_col='date')

        df_raw = df_raw.to_numpy().reshape((-1, 24, df_raw.shape[-1]))

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.lookback_len, len(df_raw) - num_test - self.lookback_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        data = df_raw[:, :, :]

        self.data_stamp = pd.date_range('2012-01-01', '2014-12-31', freq='D').to_series()[border1:border2].reset_index(
            drop=True)

        if self.scale:
            scale_data = data[:border2s[0], :, :]
            self.scaler.fit(scale_data.reshape((-1, data.shape[-1])))
            data = self.scaler.transform(data)

        if self.inverse:
            self.data_y = df_raw[border1:border2]
        else:
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        date = self.data_stamp[index].date()

        start = date - timedelta(days=self.pre_len)
        end = date + timedelta(days=self.lookback_len+self.horizon_len+self.post_len-1)
        x_date = self.data_x[start:end].values
        assert len(x_date) == self.total_len
        x_lookback = self.data_y[index:index + self.lookback_len]
        y_horizon = self.data_y[index + self.lookback_len:index + self.lookback_len + self.horizon_len]

        return x_date, x_lookback, y_horizon

    def __len__(self):
        return len(self.data_y)-self.lookback_len-self.horizon_len+1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_traffic(Dataset):
    def __init__(self, root_path, flag='train', features=None, size=None,
                 data_path=['usa_sanfrancisco.csv', 'traffic.csv'],target=None,scale=True, inverse=False,**kwargs):
        # size [pre_len, lookback_len, horizon_len, post_len]
        # info
        if size == None:
            self.pre_len = 31
            self.lookback_len = 7
            self.horizon_len = 1
            self.post_len = 31
        else:
            self.pre_len = size[0]
            self.lookback_len = size[1]
            self.horizon_len = size[2]
            self.post_len = size[3]
        self.total_len = self.pre_len + self.lookback_len + self.horizon_len + self.post_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target

        self.scale = scale
        self.inverse = inverse

        self.root_path = root_path
        self.data_path = data_path

        self.__load_data__()

    def __load_data__(self):

        self.scaler = StandardScaler()
        self.data_x = pd.read_csv(os.path.join(os.path.split(self.root_path)[0],
                                               self.data_path[0]), index_col='date', parse_dates=True,
                                  usecols=self.features)

        if self.target:
            self.target = ['date', self.target]

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path[1]),usecols=self.target,index_col='date')

        df_raw = df_raw.to_numpy().reshape((-1, 24, df_raw.shape[-1]))

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.lookback_len, len(df_raw) - num_test - self.lookback_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        data = df_raw[:, :, :]

        self.data_stamp = pd.date_range('2015-01-01', '2016-12-31', freq='D').to_series()[border1:border2].reset_index(
            drop=True)

        if self.scale:
            scale_data = data[:border2s[0], :, :]
            self.scaler.fit(scale_data.reshape((-1, data.shape[-1])))
            data = self.scaler.transform(data)

        if self.inverse:
            self.data_y = df_raw[border1:border2]
        else:
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        date = self.data_stamp[index].date()
        start = date - timedelta(days=self.pre_len)
        end = date + timedelta(days=self.lookback_len+self.horizon_len+self.post_len-1)
        x_date = self.data_x[start:end].values
        assert len(x_date) == self.total_len
        x_lookback = self.data_y[index:index + self.lookback_len]
        y_horizon = self.data_y[index + self.lookback_len:index + self.lookback_len + self.horizon_len]

        return x_date, x_lookback, y_horizon

    def __len__(self):
        return len(self.data_y)-self.lookback_len-self.horizon_len+1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_weather(Dataset):
    def __init__(self, root_path, flag='train', features=None, size=None,
                 data_path=['germany.csv', 'weather.csv'],target=None,scale=True, inverse=False,**kwargs):
        # size [pre_len, lookback_len, horizon_len, post_len]
        # info
        if size == None:
            self.pre_len = 31
            self.lookback_len = 7
            self.horizon_len = 1
            self.post_len = 31
        else:
            self.pre_len = size[0]
            self.lookback_len = size[1]
            self.horizon_len = size[2]
            self.post_len = size[3]
        self.total_len = self.pre_len + self.lookback_len + self.horizon_len + self.post_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse

        self.root_path = root_path
        self.data_path = data_path

        self.__load_data__()

    def __load_data__(self):
        self.scaler = StandardScaler()
        self.data_x = pd.read_csv(os.path.join(os.path.split(self.root_path)[0],
                                               self.data_path[0]), index_col='date', parse_dates=True,
                                  usecols=self.features)

        if self.target:
            self.target = ['date', self.target]

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path[1]),usecols=self.target,index_col='date')

        df_raw = df_raw.to_numpy().reshape((-1, 24*6, df_raw.shape[-1]))

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.lookback_len, len(df_raw) - num_test - self.lookback_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        data = df_raw[:, :, :]

        self.data_stamp = pd.date_range('2020-01-01', '2020-12-31', freq='D').to_series()[border1:border2].reset_index(
            drop=True)

        if self.scale:
            scale_data = data[:border2s[0], :, :]
            self.scaler.fit(scale_data.reshape((-1, data.shape[-1])))
            data = self.scaler.transform(data)

        if self.inverse:
            self.data_y = df_raw[border1:border2]
        else:
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        date = self.data_stamp[index].date()
        start = date - timedelta(days=self.pre_len)
        end = date + timedelta(days=self.lookback_len+self.horizon_len+self.post_len-1)
        x_date = self.data_x[start:end].values
        assert len(x_date) == self.total_len
        x_lookback = self.data_y[index:index + self.lookback_len]
        y_horizon = self.data_y[index + self.lookback_len:index + self.lookback_len + self.horizon_len]

        return x_date, x_lookback, y_horizon

    def __len__(self):
        return len(self.data_y)-self.lookback_len-self.horizon_len+1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_load(Dataset):
    def __init__(self, root_path, flag='train', features=None, size=None,
                 data_path=['china.csv', 'area1_load_15m.csv','area2_load_15m.csv'],target=None,scale=True, inverse=False,**kwargs):
        # size [pre_len, lookback_len, horizon_len, post_len]
        # info
        if size == None:
            self.pre_len = 31
            self.lookback_len = 7
            self.horizon_len = 1
            self.post_len = 31
        else:
            self.pre_len = size[0]
            self.lookback_len = size[1]
            self.horizon_len = size[2]
            self.post_len = size[3]
        self.total_len = self.pre_len + self.lookback_len + self.horizon_len + self.post_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse

        self.root_path = root_path
        self.data_path = data_path

        self.__load_data__()

    def __load_data__(self):

        self.scaler = StandardScaler()
        self.data_x = pd.read_csv(os.path.join(os.path.split(self.root_path)[0],
                                               self.data_path[0]), index_col='date', parse_dates=True,
                                  usecols=self.features)

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path[1]),usecols=self.target,index_col='date').to_numpy()
        df_raw2 = pd.read_csv(os.path.join(self.root_path, self.data_path[2]), usecols=self.target, index_col='date').to_numpy()

        num_train = int(len(df_raw) * 0.6)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.lookback_len, len(df_raw) - num_test - self.lookback_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        data = np.stack((df_raw,df_raw2),axis=-1)

        if self.target:
            data = data[:,:,0:1]

        self.data_stamp = pd.date_range('2009-01-01', '2015-01-10', freq='D').to_series()[border1:border2].reset_index(
            drop=True)

        if self.scale:
            scale_data = data[:border2s[0], :, :]
            self.scaler.fit(scale_data.reshape((-1, data.shape[-1])))
            data = self.scaler.transform(data)

        if self.inverse:
            self.data_y = self.scaler.inverse_transform(data[border1:border2])
        else:
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        date = self.data_stamp[index].date()
        start = date - timedelta(days=self.pre_len)
        end = date + timedelta(days=self.lookback_len+self.horizon_len+self.post_len-1)
        x_date = self.data_x[start:end].values
        assert len(x_date) == self.total_len
        x_lookback = self.data_y[index:index + self.lookback_len]
        y_horizon = self.data_y[index + self.lookback_len:index + self.lookback_len + self.horizon_len]

        return x_date, x_lookback, y_horizon

    def __len__(self):
        return len(self.data_y)-self.lookback_len-self.horizon_len+1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ER(Dataset):
    def __init__(self, root_path, flag='train', features=None, size=None,
                 data_path=['china.csv', 'australia.csv','british.csv','canada.csv','japan.csv','newzealand.csv','singapore.csv',
                            'switzerland.csv','usa.csv','exchange_rate.csv'],
                target=None,scale=True, inverse=False,**kwargs):
        # size [pre_len, lookback_len, horizon_len, post_len]
        # info
        if size == None:
            self.pre_len = 31
            self.lookback_len = 7
            self.horizon_len = 1
            self.post_len = 31
        else:
            self.pre_len = size[0]
            self.lookback_len = size[1]
            self.horizon_len = size[2]
            self.post_len = size[3]
        self.total_len = self.pre_len + self.lookback_len + self.horizon_len + self.post_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        assert features is not None
        self.target = target

        self.scale = scale
        self.inverse = inverse

        self.root_path = root_path
        self.data_path = data_path

        self.__load_data__()

    def __load_data__(self):
        self.scaler = StandardScaler()

        globall=self.features[0][:]; localized = self.features[1][:]
        globall.extend(localized)

        china = pd.read_csv(os.path.join(os.path.split(self.root_path)[0],
                                               self.data_path[0]), index_col='date', parse_dates=True,
                                  usecols=localized)
        australia = pd.read_csv(os.path.join(os.path.split(self.root_path)[0],
                                         self.data_path[1]), index_col='date', parse_dates=True,
                            usecols=localized)
        british = pd.read_csv(os.path.join(os.path.split(self.root_path)[0],
                                         self.data_path[2]), index_col='date', parse_dates=True,
                            usecols=localized)
        canada = pd.read_csv(os.path.join(os.path.split(self.root_path)[0],
                                         self.data_path[3]), index_col='date', parse_dates=True,
                            usecols=localized)
        japan = pd.read_csv(os.path.join(os.path.split(self.root_path)[0],
                                         self.data_path[4]), index_col='date', parse_dates=True,
                            usecols=localized)
        newzealand = pd.read_csv(os.path.join(os.path.split(self.root_path)[0],
                                         self.data_path[5]), index_col='date', parse_dates=True,
                            usecols=localized)
        singapore = pd.read_csv(os.path.join(os.path.split(self.root_path)[0],
                                              self.data_path[6]), index_col='date', parse_dates=True,
                                 usecols=localized)
        switzerland = pd.read_csv(os.path.join(os.path.split(self.root_path)[0],
                                              self.data_path[7]), index_col='date', parse_dates=True,
                                 usecols=localized)
        usa = pd.read_csv(os.path.join(os.path.split(self.root_path)[0],
                                              self.data_path[8]), index_col='date', parse_dates=True,
                                 usecols=globall)
        self.data_x = pd.concat([usa,china,australia,british,canada,japan,newzealand,singapore,switzerland],axis=1)

        if self.target:
            self.target = ['date', self.target]

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path[9]),usecols=self.target,index_col='date').to_numpy()

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.lookback_len, len(df_raw) - num_test - self.lookback_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        data = df_raw.reshape((-1,1,df_raw.shape[-1]))

        self.data_stamp = pd.date_range('2001-03-26', '2022-03-25', freq='D').to_series()[border1:border2].reset_index(
            drop=True)

        if self.scale:
            scale_data = data[:border2s[0], :, :]
            self.scaler.fit(scale_data.reshape((-1, data.shape[-1])))
            data = self.scaler.transform(data)

        if self.inverse:
            self.data_y = self.scaler.inverse_transform(data[border1:border2])
        else:
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        date = self.data_stamp[index].date()
        start = date - timedelta(days=self.pre_len)
        end = date + timedelta(days=self.lookback_len+self.horizon_len+self.post_len-1)
        x_date = self.data_x[start:end].values
        assert len(x_date) == self.total_len
        x_lookback = self.data_y[index:index + self.lookback_len]
        y_horizon = self.data_y[index + self.lookback_len:index + self.lookback_len + self.horizon_len]

        return x_date, x_lookback, y_horizon

    def __len__(self):
        return len(self.data_y)-self.lookback_len-self.horizon_len+1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

