export CUDA_VISIBLE_DEVICES=1


#load pretrained DERT encoder and train Dateformer
python -u run.py --mode 1 --pretrain both --data Weather --root_path ./data/weather --lookback_window 7 --lookback_len 7 --horizon_len 1 --train_scale 7 1 --val_scale 7 1 --d_sequences 21 --d_model 512 --e_layers 2 4 --d_layers 4 --n_heads 8 --d_ff 2048 --n_predays 7 --n_postdays 14 --granularity 144 --dropout 0.1 --prenorm --features date abs_days year day year_day lunar_year lunar_month lunar_day lunar_year_day dayofyear dayofmonth monthofyear dayoflunaryear dayoflunarmonth monthoflunaryear jieqiofyear jieqi_day dayofjieqi --test_scale 7 1 --test_scale 9 3 --test_scale 12 7 --test_scale 26 30 --des 'pretrained'
  # skip warm up
python -u run.py --mode 1 --pretrain both --data Weather --root_path ./data/weather --lookback_window 7 --lookback_len 7 --horizon_len 1 --train_scale 7 1 --val_scale 7 1 --d_sequences 21 --d_model 512 --e_layers 2 4 --d_layers 4 --n_heads 8 --d_ff 2048 --n_predays 7 --n_postdays 14 --granularity 144 --dropout 0.1 --skip_warmup --prenorm --features date abs_days year day year_day lunar_year lunar_month lunar_day lunar_year_day dayofyear dayofmonth monthofyear dayoflunaryear dayoflunarmonth monthoflunaryear jieqiofyear jieqi_day dayofjieqi --test_scale 7 1 --test_scale 9 3 --test_scale 12 7 --test_scale 26 30 --des 'pretrained without warmup'


#train Dateformer from scratch
python -u run.py --mode 1 --data Weather --root_path ./data/weather --lookback_window 7 --lookback_len 7 --horizon_len 1 --train_scale 7 1 --val_scale 7 1 --d_sequences 21 --d_model 512 --e_layers 2 4 --d_layers 4 --n_heads 8 --d_ff 2048 --n_predays 7 --n_postdays 14 --granularity 144 --dropout 0.1 --prenorm --from_scratch --features date abs_days year day year_day lunar_year lunar_month lunar_day lunar_year_day dayofyear dayofmonth monthofyear dayoflunaryear dayoflunarmonth monthoflunaryear jieqiofyear jieqi_day dayofjieqi --test_scale 7 1 --test_scale 9 3 --test_scale 12 7 --test_scale 26 30 --des 'from scratch'
  # skip warm up
python -u run.py --mode 1 --data Weather --root_path ./data/weather --lookback_window 7 --lookback_len 7 --horizon_len 1 --train_scale 7 1 --val_scale 7 1 --d_sequences 21 --d_model 512 --e_layers 2 4 --d_layers 4 --n_heads 8 --d_ff 2048 --n_predays 7 --n_postdays 14 --granularity 144 --dropout 0.1 --skip_warmup --from_scratch --prenorm  --features date abs_days year day year_day lunar_year lunar_month lunar_day lunar_year_day dayofyear dayofmonth monthofyear dayoflunaryear dayoflunarmonth monthoflunaryear jieqiofyear jieqi_day dayofjieqi --test_scale 7 1 --test_scale 9 3 --test_scale 12 7 --test_scale 26 30 --des 'from scratch without warmup'


#pretrain DERT encoder
python -u run.py --mode 0 --pretrain both --data Weather --root_path ./data/weather --batch_size 8192 --d_sequences 21 --d_model 512 --e_layers 2 4 --d_layers 4 --n_heads 8 --d_ff 2048 --n_predays 7 --n_postdays 14 --granularity 144 --dropout 0.1 --prenorm --features date abs_days year day year_day lunar_year lunar_month lunar_day lunar_year_day dayofyear dayofmonth monthofyear dayoflunaryear dayoflunarmonth monthoflunaryear jieqiofyear jieqi_day dayofjieqi --des pretrain
