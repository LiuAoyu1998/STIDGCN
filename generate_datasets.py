import argparse
import numpy as np
import os


def generate_graph_seq2seq_io_data(data, x_offsets, y_offsets):
    """
    生成seq2seq样本数据
    :param data: np数据 [B, N, D] 其中D为3
    :param x_offsets:
    :param y_offsets:
    :return:
    """
    num_samples, num_nodes, _ = data.shape
    data = data[:, :, 0:1]  # 只取第一维度的特征

    x, y = [], []
    min_t = abs(min(x_offsets))  # 11
    max_t = abs(num_samples - abs(max(y_offsets)))  # num_samples - 12

    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]  # t = 11时, [0,1,2,...,11]
        y_t = data[t + y_offsets, ...]  # t = 11时, [12,13,...,23]
        x.append(x_t)
        y.append(y_t)

    x = np.stack(x, axis=0)  # [B, T, N ,C]
    y = np.stack(y, axis=0)  # [B ,T, N, C]

    return x, y


def generate_train_val_test(args):
    """生成数据"""
    data_seq = np.load(args.traffic_df_filename)['data']
    # 交通数据 (sequence_length, num_of_vertices, num_of_features)

    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y

    x_offsets = np.arange(-(seq_length_x - 1), 1, 1)
    y_offsets = np.arange(args.y_start, (seq_length_y + 1), 1)

    x, y = generate_graph_seq2seq_io_data(
        data=data_seq, x_offsets=x_offsets, y_offsets=y_offsets)

    # [B, T, N ,C]
    print("x shape: ", x.shape, ", y shape: ", y.shape)

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.6)
    num_val = num_samples - num_train - num_test

    # 训练集
    x_train, y_train = x[:num_train], y[:num_train]

    # 验证集
    x_val, y_val = x[num_train:num_train +
                     num_val], y[num_train:num_train + num_val]

    # 测试集
    x_test, y_test = x[num_train + num_val:], y[num_train + num_val:]

    for cat in ['train', 'val', 'test']:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        # local() 是当前def中的所有变量构成的字典

        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            # 保存多个数组，按照你定义的key字典保存，compressed表示它是一个压缩文件
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            # shape从原来的(12,) 转为(12,1)
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            # shape从原来的(12,) 转为(12,1)
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        default="data/processed/PEMS08/", help="输出文件夹")
    parser.add_argument('--traffic_df_filename', type=str,
                        default="data/PEMS08/PEMS08.npz", help="数据集")
    parser.add_argument('--seq_length_x', type=int, default=12, help='输入序列长度')
    parser.add_argument('--seq_length_y', type=int, default=12, help='输出序列长度')
    parser.add_argument('--y_start', type=int, default=1, help='从第几天开始预测')

    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        reply = str(
            input(f'{args.output_dir} 存在，是否将其作为输出目录?(y/n)')).lower().strip()
        if reply[0] != 'y':
            exit()
    else:
        os.makedirs(args.output_dir)

    generate_train_val_test(args)


"""
exmaple:
PEMS04:
x shape:  (16969, 12, 307, 1) , y shape:  (16969, 12, 307, 1)
train x:  (10181, 12, 307, 1) y: (10181, 12, 307, 1)
val x:  (3394, 12, 307, 1) y: (3394, 12, 307, 1)
test x:  (3394, 12, 307, 1) y: (3394, 12, 307, 1)
PEMSO8:
x shape:  (17833, 12, 170, 1) , y shape:  (17833, 12, 170, 1)
train x:  (10700, 12, 170, 1) y: (10700, 12, 170, 1)
val x:  (3566, 12, 170, 1) y: (3566, 12, 170, 1)
test x:  (3567, 12, 170, 1) y: (3567, 12, 170, 1)
"""
