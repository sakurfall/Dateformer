import os
import time


import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data.data_loader import Dataset_ETT,  Dataset_ECL, Dataset_traffic, Dataset_weather, Dataset_load, Dataset_ER
from exp.exp_basic import Exp_Basic
from models.dert import DertforPretrain, DertWithMeanHeadforPretrain, DertWithCorrelateHeadforPretrain

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

    def __init__(self, args):
        super(Exp_DERT, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'both': DertforPretrain,
            'mean': DertWithMeanHeadforPretrain,
            'core': DertWithCorrelateHeadforPretrain,
        }
        model = model_dict[self.args.pretrain](
            d_features=self.args.d_features,
            d_sequences=self.args.d_sequences,
            d_model=self.args.d_model,
            e_layers=self.args.e_layers[0],
            n_heads=self.args.n_heads,
            d_ff=self.args.d_ff,
            n_predays=self.args.n_predays,
            attn=self.args.attn,
            factor=self.args.factor,
            activation=self.args.activation,
            output_attention=self.args.output_attention,
            prenorm=self.args.prenorm,
            dropout=self.args.dropout,
            device=self.device
        ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model



    def _test_data(self, flag):
        assert flag in ['test', 'pred']
        args = self.args
        Data = data_dict[args.data]
        if args.localized:
            features = [args.features,args.localized]
        else: features = args.features

        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            size=[args.n_predays, 0, 1, args.n_postdays],
            features=features,
            target=args.target,
            granularity=args.granularity,
            data_index=args.data_index
        )

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers)

        return data_set, data_loader

    def _train_valid_data(self, flag):
        assert flag in ['train','val']
        args = self.args
        Data = data_dict[args.data]

        if flag == 'train':
            shuffle = True
        else:
            shuffle = False

        if args.localized:
            features = [args.features,args.localized]
        else: features = args.features

        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            size=[args.n_predays, 0, 1, args.n_postdays],
            features=features,
            target=args.target,
            granularity=args.granularity,
            data_index=args.data_index
        )

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim


    def vali(self, vali_data, vali_loader):
        self.model.eval()
        total_loss = []
        for i, (x_date, _, y_horizon) in enumerate(vali_loader):
            x_date = x_date.float().to(self.device)
            y_horizon = y_horizon.squeeze(1).float().to(self.device)

            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    loss = self.model(x_date, y_horizon)
            else:
                    loss = self.model(x_date, y_horizon)

            total_loss.append(loss.detach().cpu().item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, path, test=False):
        train_data, train_loader = self._train_valid_data(flag='train')
        vali_data, vali_loader = self._train_valid_data(flag='val')
        test_data, test_loader = self._test_data(flag='test')

        print('train:', len(train_data), len(train_loader))
        print('valid:', len(vali_data), len(vali_loader))
        print('test:', len(test_data), len(test_loader))

        path = os.path.join(self.args.checkpoints, path)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(model_optim, max_lr=self.args.learning_rate, epochs=self.args.train_epochs,
                                                        steps_per_epoch=train_steps
                                                        , pct_start=0.4, three_phase=True, verbose=False)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (x_date, _, y_horizon) in enumerate(train_loader):
                iter_count += 1

                x_date = x_date.float().to(self.device)
                y_horizon = y_horizon.squeeze(1).float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        loss = self.model(x_date, y_horizon)
                else:
                    loss = self.model(x_date, y_horizon)

                train_loss.append(loss.item())

                if (i + 1) % 20 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                model_optim.zero_grad()
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            early_stopping(vali_loss, self.model, path)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
            scheduler.step()

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        if test:
            test_loss = self.vali(test_data, test_loader)
            print('Pretrained Test Loss: {:.7f}'.format(test_loss))

        return self.model





