import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sktime.datasets import load_from_tsfile_to_dataframe

import warnings
warnings.filterwarnings('ignore')

from timefeatures import time_features
from m4 import M4Dataset, M4Meta
from uea import subsample, interpolate_missing, Normalizer


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv', date_col = 'date',
                 train_test_split_rate=None, train_test_split_date=None, 
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.date_col = date_col
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # train_test_split [train_len, test_len]
        self.train_test_split_rate = train_test_split_rate
        self.train_test_split_date = train_test_split_date

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: [date features, ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)

        cols.remove(self.date_col)
        df_raw = df_raw[[self.date_col] + cols + [self.target]]

        if self.train_test_split_date != None:
            num_train = len(df_raw[df_raw[self.date_col] <= self.train_test_split_date[0]])
            num_test = len(df_raw[df_raw[self.date_col] >= self.train_test_split_date[1]])
            num_vali = len(df_raw) - num_train - num_test
        elif self.train_test_split_rate != None:
            num_train = int(len(df_raw) * self.train_test_split_rate[0])
            num_test = int(len(df_raw) * self.train_test_split_rate[1])
            num_vali = len(df_raw) - num_train - num_test
        else:
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[[self.date_col]][border1:border2]
        df_stamp[self.date_col] = pd.to_datetime(df_stamp[self.date_col])

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[self.date_col].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp[self.date_col].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp[self.date_col].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp[self.date_col].apply(lambda row: row.hour, 1)

            data_stamp = df_stamp.drop([self.date_col], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[self.date_col].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
