# Spatial-Temporal Interactive Dynamic Graph Convolution Network for Traffic Forecasting

This is the original pytorch implementation of STIDGCN in the following paper ⚡️ : [Spatial-Temporal Interactive Dynamic Graph Convolution Network for Traffic Forecasting, 2022.05.](https://arxiv.org/abs/2205.08689) I hope these codes are helpful to you, 🌟!

## Abstract

Accurate traffic forecasting is essential for urban traffic control, route planning, and flow detection. Although many spatial-temporal methods are currently proposed, they are still deficient in synchronously capturing the spatial-temporal dependence of traffic data. In addition, most methods ignore the hidden dynamic associations that arise between the road network nodes as it evolves over time. We propose a neural network-based Spatial-Temporal Interactive Dynamic Graph Convolutional Network (STIDGCN) to address the above challenges for traffic forecasting. Specifically, we propose an interactive dynamic graph convolution structure which divides the traffic data by intervals and synchronously captures the divided traffic data‘s spatial-temporal dependence through an interactive learning strategy. The interactive learning strategy motivates STIDGCN effective for long-range forecasting. We also propose a dynamic graph convolution module through a novel dynamic graph generation method to capture the dynamically changing spatial correlations in the traffic network. Based on a priori knowledge and input data, the dynamic graph generation method can generate a dynamic graph structure, which allows exploring the unseen node connections in the road network and simulating the dynamic associations between nodes over time. Extensive experiments on four real-world traffic flow datasets demonstrate that STIDGCN outperforms the state-of-the-art baselines.

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
nohup python -u train.py --data PEMS03 > PEMS03.log 2>&1 &

# PEMS04
nohup python -u train.py --data PEMS04 > PEMS04.log 2>&1 &

# PEMS07
nohup python -u train.py --data PEMS07 > PEMS07.log 2>&1 &

# PEMS08
nohup python -u train.py --data PEMS08 > PEMS08.log 2>&1 &
```
## Results

<p align="center">
  <img width="856" height="368" src=./figs/fig2.png>
</p>

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
