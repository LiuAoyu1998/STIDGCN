# STIDGCN

This is the pytorch implementation of STIDGCN. I hope these codes are helpful to you, 🌟!

## Requirements
- python
- numpy
- pandas
- torch
- matplotlib
- scipy
- argparse

## Data Preparation
Download the dataset(PEMS03, PEMS04, PEMS07, PEMS08) from here, [Baidu Drive](https://pan.baidu.com/s/1pbRUmRg_Y69KRNEuKZParQ), and the password is <b>1s5t</b>. You can put them in the "data" folder. The data here is generated using <b>generate_datasets.py</b> and <b>gen_adj_mx.py</b>, you don't need to do any further processing. If you want to see the details of how the data is processed, check out <b>generate_datasets.py</b> and <b>gen_adj_mx.py</b>.

## Train Commands

```
# PEMS03
nohup python -u train.py --data PEMS03 > PEMS03.log 2>&1 &

# PEMS04
nohup python -u train.py --data PEMS04 > PEMS04.log 2>&1 &

# PEMS07
nohup python -u train.py --data PEMS07 > PEMS07.log 2>&1 &

# PEMS08
nohup python -u train.py --data PEMS08 > PEMS08.log 2>&1 &
```

## Acknowledgments
Our model is built based on model of [Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet) and [SCINet](https://github.com/cure-lab/SCINet).
