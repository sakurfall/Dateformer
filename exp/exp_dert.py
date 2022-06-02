import os
import time


import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data.data_loader import Dataset_ETT,  Dataset_ECL, Dataset_traffic, Dataset_weather, Dataset_load, Dataset_ER
from exp.exp_basic import Exp_Basic

from utils.tools import EarlyStopping

data_dict = {
            'ETTh1': Dataset_ETT,
            'ETTh2': Dataset_ETT,
            'ETTm1': Dataset_ETT,
            'ETTm2': Dataset_ETT,
            'ECL': Dataset_ECL,
            'Traffic': Dataset_traffic,
            'Weather': Dataset_weather,
            'PowerLoad': Dataset_load,
            'ExchangeRate':Dataset_ER}

class Exp_DERT(Exp_Basic):

    pass





