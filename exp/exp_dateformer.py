import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from einops import rearrange
from data.data_loader import Dataset_ETT, Dataset_ECL, Dataset_traffic, Dataset_weather, Dataset_load, Dataset_ER
from exp.exp_basic import Exp_Basic
from models.dert import Dert
from models.model import Dateformer
from utils.metrics import metric
from utils.tools import EarlyStopping, visual, adjust_learning_rate

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



class Exp_Dateformer(Exp_Basic):

    def __init__(self, args):
        super(Exp_Dateformer, self).__init__(args)

    def _build_model(self):
        model = Dateformer(
            d_features=self.args.d_features,
            d_sequences=self.args.d_sequences,
            d_model=self.args.d_model,
            e_layers=self.args.e_layers,
            d_layers=self.args.d_layers,
            n_heads=self.args.n_heads,
            d_ff=self.args.d_ff,
            n_predays=self.args.n_predays,
            n_postdays=self.args.n_postdays,
            attn=self.args.attn,
            factor=self.args.factor,
            granularity=self.args.granularity,
            activation=self.args.activation,
            output_attention=self.args.output_attention,
            prenorm=self.args.prenorm,
            time_mapper=None if self.args.speedup else 'share',
            dropout=self.args.dropout,
            max_len=self.args.max_len,
            device=self.device
        ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag, lookback_len, horizon_len):
        assert flag in ['train', 'val', 'test']
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
            size=[args.n_predays, lookback_len, horizon_len, args.n_postdays],
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


    def _select_optimizer(self, param=None, lr=None, wd=None):
        optimizer = optim.AdamW(param, lr=lr, weight_decay=wd)
        return optimizer

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (x_date, x_lookback, y_horizon) in enumerate(vali_loader):
            x_date = x_date.float().to(self.device)
            x_lookback = x_lookback.float().to(self.device)

            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(x_date, x_lookback, self.args.lookback_window)[0]
                    else:
                        outputs = self.model(x_date, x_lookback, self.args.lookback_window)
            else:
                if self.args.output_attention:
                    outputs = self.model(x_date, x_lookback, self.args.lookback_window)[0]
                else:
                    outputs = self.model(x_date, x_lookback, self.args.lookback_window)

            loss = criterion(outputs.detach().cpu(), y_horizon)
            total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def vali_Dert(self, vali_data, vali_loader, criterion):
        self.dert.eval()
        total_loss = []
        for i, (x_date, _, y_horizon) in enumerate(vali_loader):
            x_date = x_date.float().to(self.device)
            y_horizon = y_horizon.squeeze(1).float()

            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.dert(x_date)[0]
                    else:
                        outputs = self.dert(x_date)
            else:
                if self.args.output_attention:
                    outputs = self.dert(x_date)[0]
                else:
                    outputs = self.dert(x_date)

            loss = criterion(outputs.detach().cpu(), y_horizon)
            total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.dert.train()
        return total_loss

    def warm_up(self, path):
        self.dert = Dert(
            d_features=self.args.d_features,
            d_sequences=self.args.d_sequences,
            d_model=self.args.d_model,
            e_layers=self.args.e_layers[0],
            n_heads=self.args.n_heads,
            d_ff=self.args.d_ff,
            n_predays=self.args.n_predays,
            attn=self.args.attn,
            factor=self.args.factor,
            granularity=self.args.granularity,
            activation=self.args.activation,
            output_attention=self.args.output_attention,
            prenorm=self.args.prenorm,
            dropout=self.args.dropout,
            device=self.device
        ).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            self.dert = nn.DataParallel(self.dert, device_ids=self.args.device_ids)
        if not self.args.from_scratch:
            pretrained_encoder_location = os.path.join(self.args.checkpoints,'{}/checkpoint.pth'.format(self.args.pretrain_path))
            state_dict = torch.load(pretrained_encoder_location)
            self.dert.load_state_dict(state_dict, strict=False)

        self.dert.to(self.device)



        train_data, train_loader = self._get_data(flag='train',lookback_len=0,horizon_len=1)
        vali_data, vali_loader = self._get_data(flag='val',lookback_len=0,horizon_len=1)
        test_data, test_loader = self._get_data(flag='test', lookback_len=0, horizon_len=1)

        print('train:', len(train_data), len(train_loader))
        print('valid:', len(vali_data), len(vali_loader))
        print('test:', len(test_data), len(test_loader))

        path = os.path.join(self.args.checkpoints, path)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, checkpoint_name="Dert_")

        exclude_param = []
        if not self.args.from_scratch:
            params = self._finetune_param(self.dert, state_dict)
            finetune_optim = self._select_optimizer(params, lr=self.args.finetune_rate,
                                                    wd=self.args.finetune_wd)
            exclude_param = list(map(id, params))

        optim = self._select_optimizer(filter(lambda p : id(p) not in exclude_param, self.dert.parameters()),
                                             lr=self.args.learning_rate, wd=self.args.weight_decay)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=self.args.learning_rate, epochs=self.args.train_epochs,
                                                        steps_per_epoch=train_steps
                                                        , pct_start=0.4, three_phase=True, verbose=False)
        criterion = self._select_criterion()

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
                        if self.args.output_attention:
                            outputs = self.dert(x_date)[0]
                        else:
                            outputs = self.dert(x_date)
                else:
                    if self.args.output_attention:
                        outputs = self.dert(x_date)[0]
                    else:
                        outputs = self.dert(x_date)
                assert outputs.size() == y_horizon.size()
                loss = criterion(outputs, y_horizon)
                train_loss.append(loss.item())

                if (i + 1) % 20 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                optim.zero_grad()
                if len(exclude_param)>0:
                    finetune_optim.zero_grad()
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    if len(exclude_param)>0:
                        scaler.step(finetune_optim)
                    scaler.update()
                else:
                    loss.backward()
                    optim.step()
                    if len(exclude_param)>0:
                        finetune_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali_Dert(vali_data, vali_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            early_stopping(vali_loss, self.dert, path)

            scheduler.step()
            if epoch > self.args.train_epochs*0.5 and len(exclude_param) > 0:
                adjust_learning_rate(finetune_optim, epoch-self.args.train_epochs*0.4, self.args)

        best_model_path = path + '/' + 'Dert_checkpoint.pth'
        self.dert.load_state_dict(torch.load(best_model_path))


        test_loss = self.vali_Dert(test_data, test_loader, criterion)
        print('Test Loss of warm up: {:.7f}'.format(test_loss))
        self.model.load_state_dict(torch.load(best_model_path),strict=False)
        return self.model



    def train(self, path):
        lookback_len, horizon_len = self.args.train_scale
        train_data, train_loader = self._get_data(flag='train',lookback_len=lookback_len,horizon_len=horizon_len)
        lookback_len, horizon_len = self.args.val_scale
        vali_data, vali_loader = self._get_data(flag='val',lookback_len=lookback_len,horizon_len=horizon_len)

        print('train:', len(train_data), len(train_loader))
        print('valid:', len(vali_data), len(vali_loader))

        path = os.path.join(self.args.checkpoints, path)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, checkpoint_name="Dateformer_")
        exclude_param = []
        if not self.args.skip_warmup:
            state_dict = torch.load(path + '/' + 'Dert_checkpoint.pth')
            params = self._finetune_param(self.model, state_dict)
            finetune_optim = self._select_optimizer(params, lr=self.args.finetune_rate, wd=self.args.finetune_wd)
            exclude_param = list(map(id, params))
        elif not self.args.from_scratch:
            pretrained_encoder_location = os.path.join(self.args.checkpoints,
                                                       '{}/checkpoint.pth'.format(self.args.pretrain_path))
            state_dict = torch.load(pretrained_encoder_location)
            self.model.load_state_dict(state_dict, strict=False)
            params = self._finetune_param(self.model, state_dict)
            finetune_optim = self._select_optimizer(params, lr=self.args.finetune_rate, wd=self.args.finetune_wd)
            exclude_param = list(map(id, params))


        model_optim = self._select_optimizer(filter(lambda p : id(p) not in exclude_param, self.model.parameters()),
                                             lr=self.args.learning_rate, wd=self.args.weight_decay)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(model_optim, max_lr=self.args.learning_rate, epochs=self.args.train_epochs,
                                                        steps_per_epoch=train_steps
                                                        , pct_start=0.4, three_phase=True, verbose=False)
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (x_date, x_lookback, y_horizon) in enumerate(train_loader):
                iter_count += 1

                x_date = x_date.float().to(self.device)
                x_lookback = x_lookback.float().to(self.device)
                y_horizon = y_horizon.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(x_date, x_lookback, self.args.lookback_window)[0]
                        else:
                            outputs = self.model(x_date, x_lookback, self.args.lookback_window)
                else:
                    if self.args.output_attention:
                        outputs = self.model(x_date, x_lookback, self.args.lookback_window)[0]
                    else:
                        outputs = self.model(x_date, x_lookback, self.args.lookback_window)

                loss = criterion(outputs, y_horizon)
                train_loss.append(loss.item())

                if (i + 1) % 20 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                model_optim.zero_grad()
                if len(exclude_param) > 0:
                    finetune_optim.zero_grad()
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    if len(exclude_param) > 0:
                        scaler.step(finetune_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    if len(exclude_param) > 0:
                        finetune_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            early_stopping(vali_loss, self.model, path)
            scheduler.step()
            if epoch > self.args.train_epochs*0.5 and len(exclude_param) > 0:
                adjust_learning_rate(finetune_optim, epoch-self.args.train_epochs*0.4, self.args)

        best_model_path = path + '/' + 'Dateformer_checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, lookback_len, horizon_len, f, path=None):
        test_data, test_loader = self._get_data('test', lookback_len, horizon_len)
        print('test:', len(test_data), len(test_loader))
        if path:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + path, 'Dateformer_checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/{}/{}To{}/'.format(setting,lookback_len,horizon_len)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (x_date, x_lookback, y_horizon) in enumerate(test_loader):
                x_date = x_date.float().to(self.device)
                x_lookback = x_lookback.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(x_date, x_lookback, self.args.lookback_window)[0]
                        else:
                            outputs = self.model(x_date, x_lookback, self.args.lookback_window)
                else:
                    if self.args.output_attention:
                        outputs = self.model(x_date, x_lookback, self.args.lookback_window)[0]
                    else:
                        outputs = self.model(x_date, x_lookback, self.args.lookback_window)
                pred = rearrange(outputs.detach().cpu(),'b d h w->b (d h) w').numpy()
                true = rearrange(y_horizon,'b d h w->b (d h) w').numpy()

                preds.append(pred)
                trues.append(true)
                if i % 3 == 0:
                    input = x_lookback.detach().cpu()[:,-1,:,:].numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds,axis=0)
        trues = np.concatenate(trues,axis=0)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/{}/{}To{}/'.format(setting,lookback_len,horizon_len)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('{} -> {}  mse:{}, mae:{}'.format(lookback_len,horizon_len,mse, mae))
        f.write('{} -> {}  mse:{}, mae:{}\n'.format(lookback_len,horizon_len, mse, mae))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def _finetune_param(self, model, pretrained_dict):
        params = []
        for (name, param) in model.named_parameters():
            if name in pretrained_dict:
                params.append(param)

        return params








