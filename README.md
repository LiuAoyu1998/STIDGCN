# Spatial-Temporal Interactive Dynamic Graph Convolution Network for Traffic Forecasting

This is the original pytorch implementation of STIDGCN in the following paper: 
[Spatial-Temporal Interactive Dynamic Graph Convolution Network for Traffic Forecasting, 2022] (https://arxiv.org/abs/2205.08689).



<p align="center">
  <img width="835" height="408" src=./figs/fig1.png>
</p>

## Requirements
- python
- numpy
- pandas
- torch
- matplotlib
- scipy
- argparse

## Data Preparation
Download the dataset(PEMS03, PEMS04, PEMS07, PEMS08) from here, [Aliyun Drive](https://www.aliyundrive.com/s/P3UGEdMCkh1), and the password is <b>sj35</b>. You can put them in the "data" folder. You can access the <b>generate_datasets.py</b> and <b>gen_adj_mx.py</b> files to get the dataset and adjacency matrix generation methods.

## Train Commands

```
# PEMS03
nohup python -u train.py --data PEMS03 --save ./logs/PEMS03/ > PEMS03.log 2>&1 &

# PEMS04
nohup python -u train.py --data PEMS04 --save ./logs/PEMS04/ > PEMS04.log 2>&1 &

# PEMS07
nohup python -u train.py --data PEMS07 --save ./logs/PEMS07/ > PEMS07.log 2>&1 &

# PEMS08
nohup python -u train.py --data PEMS08 --save ./logs/PEMS08/ > PEMS08.log 2>&1 &
```

## Cite
If you make use of this code in your own work, please cite our paper:
 ```latex
@misc{liu2022spatialtemporal,
      title={Spatial-Temporal Interactive Dynamic Graph Convolution Network for Traffic Forecasting}, 
      author={Aoyu Liu and Yaying Zhang},
      year={2022},
      eprint={2205.08689},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgments
Our model is built based on model of [Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet) and [SCINet](https://github.com/cure-lab/SCINet).