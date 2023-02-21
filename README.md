# Dateformer

Dateformer: Time-modeling Transformer for Long-term Series Forecasting

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Get Started

To reproduce the results in the paper, run this command:

```bash
bash ./scripts/experiments.sh
```
## Results

We experiment on 7 datasets, covering 4 main-stream applications. We compare our model with 6 baselines, including FEDformer, Autoformer, Informer, Pyraformer, etc. Generally, for the all setups on all datasets, Dateformer achieves the SOTA performance, with a **33.6% relative improvement** over previous baselines.
<p align="center">
<img src=".\pic\result.png" height = "550" alt="" align=center />
</p>

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/DAMO-DI-ML/ICML2022-FEDformer

https://github.com/zhouhaoyi/Informer2020

https://github.com/zhouhaoyi/ETDataset

https://github.com/thuml/Autoformer

https://github.com/laiguokun/multivariate-time-series-data