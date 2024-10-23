import torch
import torch.nn as nn
from torch.nn import Conv2d, Parameter
import torch.nn.functional as F
import math

class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out


class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 1]
        time_day = self.time_day[(day_emb[:, :, :] * self.time).type(torch.LongTensor)]
        time_day = time_day.transpose(1, 2).contiguous()

        week_emb = x[..., 2]
        time_week = self.time_week[(week_emb[:, :, :]).type(torch.LongTensor)]
        time_week = time_week.transpose(1, 2).contiguous()

        tem_emb = time_day + time_week

        tem_emb = tem_emb.permute(0,3,1,2)

        return tem_emb


    
class Diffusion_GCN(nn.Module):
    def __init__(self, channels=128, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.diffusion_step = diffusion_step
        self.conv = nn.Conv2d(diffusion_step * channels, channels, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        out = []
        for i in range(0, self.diffusion_step):
            if adj.dim() == 3:
                x = torch.einsum("bcnt,bnm->bcmt", x, adj).contiguous()
                out.append(x)
            elif adj.dim() == 2:
                x = torch.einsum("bcnt,nm->bcmt", x, adj).contiguous()
                out.append(x)
        x = torch.cat(out, dim=1)
        x = self.conv(x)
        output = self.dropout(x)
        return output

    
class Graph_Generator(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(channels, num_nodes))
        nn.init.xavier_uniform_(self.memory)
        self.fc = nn.Linear(2,1)
        
    def forward(self, x):
        adj_dyn_1 = torch.softmax(
            F.relu(
                torch.einsum("bcnt, cm->bnm", x, self.memory).contiguous()
                / math.sqrt(x.shape[1])
            ),
            -1,
        )
        adj_dyn_2 = torch.softmax(
            F.relu(
                torch.einsum("bcn, bcm->bnm", x.sum(-1), x.sum(-1)).contiguous()
                / math.sqrt(x.shape[1])
            ),
            -1,
        )
        # adj_dyn = (adj_dyn_1 + adj_dyn_2 + adj)/2
        adj_f = torch.cat([(adj_dyn_1).unsqueeze(-1)] + [(adj_dyn_2).unsqueeze(-1)], dim=-1)
        adj_f = torch.softmax(self.fc(adj_f).squeeze(), -1)

        topk_values, topk_indices = torch.topk(adj_f, k=int(adj_f.shape[1]*0.8), dim=-1)
        mask = torch.zeros_like(adj_f)
        mask.scatter_(-1, topk_indices, 1)
        adj_f = adj_f * mask
        
        return adj_f


class DGCN(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1, emb=None):
        super().__init__()
        self.conv = nn.Conv2d(channels,channels,(1,1))
        self.generator = Graph_Generator(channels, num_nodes, diffusion_step, dropout)
        self.gcn = Diffusion_GCN(channels, diffusion_step, dropout)
        self.emb = emb

    def forward(self, x):
        skip = x
        x = self.conv(x)
        adj_dyn = self.generator(x)
        x = self.gcn(x, adj_dyn) 
        x = x*self.emb + skip
        return x


class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, :, :, ::2]

    def odd(self, x):
        return x[:, :, :, 1::2]

    def forward(self, x):
        return (self.even(x), self.odd(x))


class IDGCN(nn.Module):
    def __init__(
        self,
        device,
        channels=64,
        diffusion_step=1,
        splitting=True,
        num_nodes=170,
        dropout=0.2, emb = None
    ):
        super(IDGCN, self).__init__()

        device = device
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.splitting = splitting
        self.split = Splitting()

        Conv1 = []
        Conv2 = []
        Conv3 = []
        Conv4 = []
        pad_l = 3
        pad_r = 3

        k1 = 5
        k2 = 3
        Conv1 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1, k1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels, kernel_size=(1, k2)),
            nn.Tanh(),
        ]
        Conv2 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1, k1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels, kernel_size=(1, k2)),
            nn.Tanh(),
        ]
        Conv4 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1, k1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels, kernel_size=(1, k2)),
            nn.Tanh(),
        ]
        Conv3 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1, k1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels, kernel_size=(1, k2)),
            nn.Tanh(),
        ]

        self.conv1 = nn.Sequential(*Conv1)
        self.conv2 = nn.Sequential(*Conv2)
        self.conv3 = nn.Sequential(*Conv3)
        self.conv4 = nn.Sequential(*Conv4)

        self.dgcn = DGCN(channels, num_nodes, diffusion_step, dropout, emb)

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        x1 = self.conv1(x_even)
        x1 = self.dgcn(x1)
        d = x_odd.mul(torch.tanh(x1))

        x2 = self.conv2(x_odd)
        x2 = self.dgcn(x2)
        c = x_even.mul(torch.tanh(x2))

        x3 = self.conv3(c)
        x3 = self.dgcn(x3)
        x_odd_update = d + x3

        x4 = self.conv4(d)
        x4 = self.dgcn(x4)
        x_even_update = c + x4

        return (x_even_update, x_odd_update)


class IDGCN_Tree(nn.Module):
    def __init__(
        self, device, channels=64, diffusion_step=1, num_nodes=170, dropout=0.1
    ):
        super().__init__()

        self.memory1 = nn.Parameter(torch.randn(channels, num_nodes, 3))

        self.IDGCN = IDGCN(
            device=device,
            splitting=True,
            channels=channels,
            diffusion_step=diffusion_step,
            num_nodes=num_nodes,
            dropout=dropout,emb=self.memory1
        )

    def concat(self, even, odd):
        even = even.permute(3, 1, 2, 0)
        odd = odd.permute(3, 1, 2, 0)
        len = even.shape[0]
        _ = []
        for i in range(len):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        return torch.cat(_, 0).permute(3, 1, 2, 0)

    def forward(self, x):
        x_even_update1, x_odd_update1 = self.IDGCN(x)
        concat = self.concat(x_even_update1, x_odd_update1)
        output = concat + x
        return output


class STIDGCN(nn.Module):
    def __init__(self, device, num_nodes, channels, dropout=0.1, pre_adj=None):
        super().__init__()

        # attributes
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = 6
        self.input_dim = 3
        self.output_len = 1
        self.num_layer = 1
        self.head = 1
        self.num_levels = 2

        self.start_conv = nn.Conv2d(
            in_channels=self.input_dim, out_channels=channels, kernel_size=(1, 1)
        )

        if num_nodes ==1024:
            self.time = 24
        else:
            self.time = 48

        self.Temb = TemporalEmbedding(self.time, channels)

        self.tree = IDGCN_Tree(
            device=device,
            channels=channels*2,
            diffusion_step=1,
            num_nodes=self.num_nodes,
            dropout=dropout,
        )

        self.glu = GLU(channels*2)
        self.regression_layer = nn.Conv2d(
            channels*2, self.output_len, kernel_size=(1, 6)
        )

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, input):
        x = input
        # Encoder
        # Data Embedding
        time_emb = self.Temb(input.permute(0, 3, 2, 1))
        x = torch.cat([self.start_conv(x)] + [time_emb], dim=1)
        # IDGCN_Tree
        x = self.tree(x)
        # Decoder
        gcn = self.glu(x) + x
        prediction = self.regression_layer(F.relu(gcn))
        return prediction
