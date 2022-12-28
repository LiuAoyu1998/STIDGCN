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

## Datasets
Download the dataset(PEMS03, PEMS04, PEMS07, PEMS08) from here, [STSGCN](https://github.com/Davidham3/STSGCN). The data is processed in the same as STSGCN.

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
