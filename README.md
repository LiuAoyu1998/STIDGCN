# STIDGCN
This is the pytorch implementation of STIDGCN. I hope these codes are helpful to you!

[STIDGCN is accepted by TITS (IEEE Transactions on Intelligent Transportation Systems).](https://ieeexplore.ieee.org/document/10440184)

<p align="center">
  <img src="figs/model.png" width="100%">
</p>

## Requirements
The code is built based on Python 3.9.12, PyTorch 1.11.0, and NumPy 1.21.2.

## Datasets
We provide preprocessed datasets that you can access [here](https://drive.google.com/drive/folders/1-5hKD4hKd0eRdagm4MBW1g5kjH5qgmHR?usp=sharing). If you need the original datasets, please refer to [STSGCN](https://github.com/Davidham3/STSGCN) (including PEMS03, PEMS04, PEMS07, and PEMS08) and [ESG](https://github.com/LiuZH-19/ESG) (including NYCBike and NYCTaxi).

## Train Commands
It's easy to run! Here are some examples, and you can customize the model settings in train.py.
### PEMS08
```
nohup python -u train.py --data PEMS08 --batch_size 64 > PEMS08.log &
```
### NYCBike Drop-off
```
nohup python -u train.py --data bike_drop --batch_size 16 > bike_drop.log &
```
### TDrive Inflow
```
nohup python -u train_grid.py --data TDrive_i --batch_size 16 > TDrive_i.log &
```

## Results
<p align="center">
<img src="figs/result_1.png" width="100%">
</p>
<p align="center">
<img src="figs/result_2.png" width="100%">
</p>
<p align="center">
<img src="figs/result_3.png" width="100%">
</p>

## Acknowledgments
Our model is built based on model of [Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet) and [SCINet](https://github.com/cure-lab/SCINet).
