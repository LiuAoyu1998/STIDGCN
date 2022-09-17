import util
import argparse
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str,
                    default='data/PEMS08', help='data path')
parser.add_argument('--adjdata', type=str,
                    default='data/PEMS08/adj_PEMS08.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str,
                    default='doubletransition', help='adj type')
parser.add_argument('--num_nodes', type=int,
                    default=170, help='number of nodes')
parser.add_argument('--channels', type=int,
                    default=64, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float,
                    default=0.0001, help='weight decay rate')
parser.add_argument('--checkpoint', type=str,
                    default='logs/test/best_model.pth', help='')
parser.add_argument('--plotheatmap', type=str, default='True', help='')
args = parser.parse_args()


def main():
    if args.data == "PEMS08":
        args.data = "data/"+args.data
        args.num_nodes = 170
        args.adjdata = "data/PEMS08/adj_PEMS08.pkl"

    elif args.data == "PEMS03":
        args.data = "data/"+args.data
        args.num_nodes = 358
        args.adjdata = "data/PEMS03/adj_PEMS03.pkl"

    elif args.data == "PEMS04":
        args.data = "data/"+args.data
        args.num_nodes = 307
        args.adjdata = "data/PEMS04/adj_PEMS04.pkl"

    elif args.data == "PEMS07":
        args.data = "data/"+args.data
        args.num_nodes = 883
        args.adjdata = "data/PEMS07/adj_PEMS07.pkl"

    device = torch.device(args.device)

    _, _, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    pre_adj = [torch.tensor(i).to(device) for i in adj_mx]

    model = STIDGCN(device, args.num_nodes, args.channels,
                    args.dropout, pre_adj=pre_adj)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    print('model load successfully')

    dataloader = util.load_dataset(
        args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    amae = []
    amape = []
    armse = []
    
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))

    if args.plotheatmap == "True":
        adp = F.softmax(
            F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
        device = torch.device('cpu')
        adp.to(device)
        adp = adp.cpu().detach().numpy()
        adp = adp*(1/np.max(adp))
        df = pd.DataFrame(adp)
        sns.heatmap(df, cmap="RdYlBu")
        plt.savefig("./emb" + '.pdf')

    y12 = realy[:, 99, 11].cpu().detach().numpy()
    yhat12 = scaler.inverse_transform(yhat[:, 99, 11]).cpu().detach().numpy()

    y3 = realy[:, 99, 2].cpu().detach().numpy()
    yhat3 = scaler.inverse_transform(yhat[:, 99, 2]).cpu().detach().numpy()

    df2 = pd.DataFrame({'real12': y12, 'pred12': yhat12,
                        'real3': y3, 'pred3': yhat3})
    df2.to_csv('./wave.csv', index=False)


if __name__ == "__main__":
    main()
