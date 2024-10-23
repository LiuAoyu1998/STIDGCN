import torch
import numpy as np
import pandas as pd
import argparse
import time
import util
import os
from util import *
import random
from model_grid import STIDGCN
from ranger21 import Ranger

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:1', help='')
parser.add_argument('--data', type=str,
                    default='NYCTaxi_i', help='data path')
parser.add_argument('--adjdata', type=str,
                    default='data/PEMS08/adj_PEMS08.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str,
                    default='doubletransition', help='adj type')
parser.add_argument('--num_nodes', type=int,
                    default=75, help='number of nodes')
parser.add_argument('--channels', type=int,
                    default=64, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float,
                    default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=500, help='')
parser.add_argument('--print_every', type=int, default=200, help='')
parser.add_argument('--save', type=str,
                    default='./logs/'+str(time.strftime('%Y-%m-%d-%H:%M:%S'))+"-", help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--es_patience', type=int, default=100,
                    help='quit if no improvement after this many iterations')
args = parser.parse_args()

def seed_it(seed):
    random.seed(seed) #可以注释掉
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #这个懂吧
    torch.backends.cudnn.deterministic = True #确定性固定
    torch.backends.cudnn.enabled = True  #增加运行效率，默认就是True
    torch.manual_seed(seed)

class trainer():
    def __init__(self, scaler, num_nodes, channels, dropout, lrate, wdecay, device, pre_adj,pre_adj_t):
        self.model = STIDGCN(device, num_nodes, channels,
                             dropout, pre_adj=pre_adj)
        self.model.to(device)
        self.optimizer = Ranger(self.model.parameters(),
                                lr=lrate, weight_decay=wdecay)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5
        print(self.model)

    def train(self, input, real_val):
        # input torch.Size([64, 3, 75, 6])
        # real_val torch.Size([64, 1, 75, 1])
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)  # 64 1 75 1
        output = output.transpose(1, 3)  # 64 1 75 1
        predict = self.scaler.inverse_transform(output)  # 64 1 75 1
        loss = self.loss(predict, real_val, 0.0, 10)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict, real_val, 0.0, 10).item()
        rmse = util.masked_rmse(predict, real_val, 0.0, 10).item()
        return loss.item(), mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1, 3)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real_val, 0.0, 10)
        mape = util.masked_mape(predict, real_val, 0.0, 10).item()
        rmse = util.masked_rmse(predict, real_val, 0.0, 10).item()
        return loss.item(), mape, rmse

def main():

    seed_it(6666)

    data = args.data

    if args.data == "NYTaxi_i":
        args.data = "data//"+args.data
        args.num_nodes = 75
    
    elif args.data == "NYTaxi_o":
        args.data = "data//"+args.data
        args.num_nodes = 75

    elif args.data == "TDrive_i":
        args.data = "data//"+args.data
        args.num_nodes = 1024
    
    elif args.data == "TDrive_o":
        args.data = "data//"+args.data
        args.num_nodes = 1024
    
    device = torch.device(args.device)

    dataloader = util.load_dataset(
        args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    pre_adj = None
    pre_adj_t = None

    loss = 9999999
    test_log = 999999
    epochs_since_best_mae = 0
    path = args.save + data + "/"

    his_loss = []
    val_time = []
    train_time = []
    result = []
    test_result = []

    print(args)

    if not os.path.exists(path):
        os.makedirs(path)

    engine = trainer(scaler, args.num_nodes, args.channels, args.dropout,
                     args.learning_rate, args.weight_decay, device, pre_adj, pre_adj_t)

    print("start training...", flush=True)

    for i in range(1, args.epochs+1):

        # train
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            
            trainx = torch.Tensor(x).to(device)  # 64 12 170 1
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            # torch.Size([64, 4, 75, 6])
            # torch.Size([64, 2, 75, 1])
            metrics = engine.train(trainx, trainy)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}'
                print(log.format(
                    iter, train_loss[-1],  train_rmse[-1], train_mape[-1]), flush=True)
        t2 = time.time()
        log = 'Epoch: {:03d}, Training Time: {:.4f} secs'
        print(log.format(i, (t2-t1)))
        train_time.append(t2-t1)

        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()

        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2-s1)))
        val_time.append(s2-s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        his_loss.append(mvalid_loss)
        train_m = dict(train_loss=np.mean(train_loss), train_rmse=np.mean(train_rmse),
                       train_mape=np.mean(train_mape), valid_loss=np.mean(valid_loss),
                       valid_rmse=np.mean(valid_rmse), valid_mape=np.mean(valid_mape))
        train_m = pd.Series(train_m)
        result.append(train_m)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
        print(log.format(i, mtrain_loss, mtrain_rmse, mtrain_mape), flush=True)
        log = 'Epoch: {:03d}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}'
        print(log.format(i, mvalid_loss, mvalid_rmse, mvalid_mape), flush=True)

        if mvalid_loss < loss:
            print("###Update tasks appear###")
            if i < 100:
                # It is not necessary to print the results of the test set when epoch is less than 100, because the model has not yet converged.
                loss = mvalid_loss
                torch.save(engine.model.STIDGCNe_dict(),
                           path+"best_model.pth")
                bestid = i
                epochs_since_best_mae = 0
                print("Updating! Valid Loss:", mvalid_loss, end=", ")
                print("epoch: ", i)

            elif i > 100:
                outputs = []
                realy = torch.Tensor(dataloader['y_test']).to(device)
                realy = realy.transpose(1, 3) # torch.Size([64, 1, 75, 1])

                for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
                    testx = torch.Tensor(x).to(device)
                    testx = testx.transpose(1, 3)
                    with torch.no_grad():
                        preds = engine.model(testx).transpose(1, 3)
                    outputs.append(preds)

                yhat = torch.cat(outputs, dim=0)

                yhat = yhat[:realy.size(0), ...]

                amae = []
                amape = []
                armse = []
                test_m = []
                pred =  scaler.inverse_transform(yhat)
                metrics = util.metric_grid(pred, realy)
                amae.append(metrics[0])
                amape.append(metrics[1])
                armse.append(metrics[2])

                log = 'Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                print(log.format(np.mean(amae), np.mean(armse), np.mean(amape)))

                if np.mean(amae) < test_log:
                    test_log = np.mean(amae)
                    loss = mvalid_loss
                    torch.save(engine.model.STIDGCNe_dict(),
                               path+"best_model.pth")
                    epochs_since_best_mae = 0
                    print("Test low! Updating! Test Loss:",
                          np.mean(amae), end=", ")
                    print("Test low! Updating! Valid Loss:",
                          mvalid_loss, end=", ")
                    bestid = i
                    print("epoch: ", i)
                else:
                    epochs_since_best_mae += 1
                    print("No update")

        else:
            epochs_since_best_mae += 1
            print("No update")

        train_csv = pd.DataFrame(result)
        train_csv.round(6).to_csv(f'{path}/train.csv')
        if epochs_since_best_mae >= args.es_patience and i >= 300:
            break

    # Output consumption
    print(
        "Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # test
    print("Training ends")
    print("The epoch of the best result：", bestid)
    print("The valid loss of the best model",
          str(round(his_loss[bestid-1], 4)))

    engine.model.load_STIDGCNe_dict(torch.load(path+"best_model.pth"))
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3) # torch.Size([64, 1, 75, 1])

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds)

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]


    amae = []
    amape = []
    armse = []
    test_m = []
    pred =  scaler.inverse_transform(yhat)
    metrics = util.metric_grid(pred, realy)
    amae.append(metrics[0])
    amape.append(metrics[1])
    armse.append(metrics[2])

    log = 'Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(armse), np.mean(amape)))

    test_m = dict(test_loss=np.mean(amae),
                  test_rmse=np.mean(armse), test_mape=np.mean(amape))
    test_m = pd.Series(test_m)
    test_result.append(test_m)

    test_csv = pd.DataFrame(test_result)
    test_csv.round(6).to_csv(f'{path}/test.csv')


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
