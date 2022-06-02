import argparse
import random

import numpy as np
import torch

from exp.exp_dateformer import Exp_Dateformer
from exp.exp_dert import Exp_DERT

fix_seed = 20210827
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='[Dateformer] Time-modeling Long-term Series Forecasting')

# basic config
parser.add_argument('--mode', type=int, required=True, help='status {0:pretrain,1:train,2:test}')
parser.add_argument('--model', type=str,  default='Dateformer',help='model of experiment')
parser.add_argument('--pretrain', type=str,  default="both", help='model for pretrain')
# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT', help='root path of the data file')
parser.add_argument('--features', nargs='+', type=str, help="choose time features from date", required=True)
parser.add_argument('--localized', nargs='+', type=str, help="choose localized time features from date", default=None)
parser.add_argument('--target', type=str, default=None, help='forecast target feature in S task, none means M task')
parser.add_argument('--data_index', type=int, default=1, help='data file index for ETT')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--pretrain_path', type=str, help='location of pretrained encoder checkpoints')
# forecasting task
parser.add_argument('--lookback_window', type=int, default=7, help='num of days in lookback window')
parser.add_argument('--lookback_len', type=int, default=7, help='num of days to predict')
parser.add_argument('--horizon_len', type=int, default=1, help='num of days to provide local information')

#train config
parser.add_argument('--train_scale', type=int, nargs=2, default=[7, 1], help='train scale')
parser.add_argument('--val_scale', type=int, nargs=2, default=[7, 1], help='valid scale')
parser.add_argument('--test_scale', type=int, nargs=2, action='append', help='test scale')

# model define
parser.add_argument('--d_sequences', type=int, default=7, help='dimension of sequences')
parser.add_argument('--d_features', type=int, default=None, help='dimension of features')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--e_layers', nargs=2,type=int, default=[1,1], help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help="num of longlongformer's decoder layers")
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--n_predays', type=int, default=7, help='num of predays')
parser.add_argument('--n_postdays', type=int, default=14, help='num of postdays')
parser.add_argument('--attn', type=str, default='full', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--speedup', action='store_true', help='whether to speed up longlongformer')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--granularity', type=int, default=96, help='num of data points in a day')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--prenorm', action='store_true')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--max_len', type=int, default=720, help="maxcount of tokens")
parser.add_argument('--from_scratch', action='store_true', help='whether to skip loading pretrained DERT encoder')
parser.add_argument('--skip_warmup', action='store_true', help='whether to skip warm up step')
# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=3, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='optimizer learning rate')
parser.add_argument('--finetune_rate', type=float, default=0, help='finetune learning rate')
parser.add_argument('--weight_decay', type=float, default=7e-4, help='optimizer weight decay')
parser.add_argument('--finetune_wd', type=float, default=0, help='optimizer weight decay')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)


#GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

args = parser.parse_args()

args.pretrain_path = args.pretrain_path or 'pretrain_{}_{}'.format(args.pretrain,args.data)
args.d_features = args.d_features or len(args.features)-1
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]


print('Args in experiment:')
print(args)

#pretrain
if args.mode == 0:
    setting = '{}_{}_gra{}_pre{}_post{}_fn{}_dm{}_nh{}_el{}_df{}_{}_{}'.format(
        args.pretrain,
        args.data,
        args.granularity,
        args.n_predays,
        args.n_postdays,
        args.d_sequences,
        args.d_model,
        args.n_heads,
        args.e_layers[0],
        args.d_ff,
        args.attn,
        args.des)
    exp = Exp_DERT(args)
    print('>>>>>>>start pretraining : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.train(args.pretrain_path,test=True)
#train
elif args.mode == 1:
    for ii in range(args.itr):
        # setting record of experiments

        setting = '{}_{}_gra{}_pre{}_post{}_fn{}_dm{}_nh{}_el{},{}_dl{}_df{}_{}_ts{},{}_vs{},{}_{}_{}'.format(
            args.model,
            args.data,
            args.granularity,
            args.n_predays,
            args.n_postdays,
            args.d_sequences,
            args.d_model,
            args.n_heads,
            args.e_layers[0],
            args.e_layers[1],
            args.d_layers,
            args.d_ff,
            args.attn,
            args.train_scale[0],
            args.train_scale[1],
            args.val_scale[0],
            args.val_scale[1],
            args.des, ii)
        path = '{}_{}_{}'.format(
            args.model,
            args.data,
            ii)
        exp = Exp_Dateformer(args)  # set experiments
        if not args.skip_warmup:
            print('>>>>>>>start warm up : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.warm_up(path)
            torch.cuda.empty_cache()

        print('>>>>>>>start training : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.train(path)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        for l, h in args.test_scale:
            torch.cuda.empty_cache()
            exp.test(setting, l, h, f)
            f.write('\n')
        f.write('\n')
        f.close()
elif args.mode == 2:
#test
    ii = 0
    setting = '{}_{}_gra{}_pre{}_post{}_fn{}_dm{}_nh{}_el{},{}_dl{}_df{}_{}_ts{},{}_vs{},{}_{}_{}'.format(
        args.model,
        args.data,
        args.granularity,
        args.n_predays,
        args.n_postdays,
        args.d_sequences,
        args.d_model,
        args.n_heads,
        args.e_layers[0],
        args.e_layers[1],
        args.d_layers,
        args.d_ff,
        args.attn,
        args.train_scale[0],
        args.train_scale[1],
        args.val_scale[0],
        args.val_scale[1],
        args.des, ii)
    path = '{}_{}_{}'.format(
        args.model,
        args.data,
        ii)
    exp = Exp_Dateformer(args)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    f = open("result.txt", 'a')
    f.write(setting + "  \n")
    torch.cuda.empty_cache()
    exp.test(setting, args.lookback_len, args.horizon_len, f, path)
    f.write('\n')
    f.write('\n')
    f.close()
else: raise NotImplementedError

