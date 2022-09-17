from model import *
import util
from ranger21 import Ranger


class trainer():
    def __init__(self, scaler, num_nodes, channels, dropout, lrate, wdecay, device, pre_adj):
        self.model = STIDGCN(device, num_nodes, channels,
                             dropout, pre_adj=pre_adj)
        self.model.to(device)
        self.optimizer = Ranger(self.model.parameters(),
                                lr=lrate, weight_decay=wdecay)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        # input 64 1 170 12
        # real_val 64 17 12
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)  # 64 12 170 1
        output = output.transpose(1, 3)  # 64 1 170 12
        real = torch.unsqueeze(real_val, dim=1)  # 64 1 170 12
        predict = self.scaler.inverse_transform(output)  # 64 1 170 12
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse
