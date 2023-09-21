import torch
import torch.nn as nn
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
        time_day = self.time_day[(day_emb[:, -1, :] * self.time).type(torch.LongTensor)]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        week_emb = x[..., 2]
        time_week = self.time_week[(week_emb[:, -1, :]).type(torch.LongTensor)]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        tem_emb = time_day + time_week
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
                x = torch.einsum("bcnt,bnm->bcmt", x, adj)
                out.append(x)
            elif adj.dim() == 2:
                x = torch.einsum("bcnt,nm->bcmt", x, adj)
                out.append(x)
        x = torch.cat(out, dim=1)
        x = self.conv(x)
        output = self.dropout(x)
        return output


class Graph_Generator(nn.Module):
    def __init__(self, channels=128, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.gcn = Diffusion_GCN(channels, diffusion_step, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = self.gcn(x, adj)
        x = self.dropout(x)
        adj_dyn = torch.softmax(
            F.relu(
                torch.einsum("bcn, bcm->bnm", x.sum(3), x.sum(3))
                / math.sqrt(x.shape[1])
            ),
            -1,
        )
        adj_dyn = adj_dyn + adj
        return adj_dyn


class DGCN(nn.Module):
    def __init__(self, channels=128, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.generator = Graph_Generator(channels, diffusion_step, dropout)
        self.gcn = Diffusion_GCN(channels, diffusion_step, dropout)

    def forward(self, x, adj):
        adj_dyn = self.generator(x, adj)
        x = self.gcn(x, adj_dyn) + x
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
        dropout=0.2,
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

        apt_size = 10
        nodevecs = torch.randn(num_nodes, apt_size), torch.randn(apt_size, num_nodes)
        self.nodevec1, self.nodevec2 = [
            nn.Parameter(n.to(device), requires_grad=True) for n in nodevecs
        ]

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

        self.dgcn = DGCN(channels, diffusion_step, dropout)

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        adaptive_adj = torch.softmax(
            F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=-1
        )

        x1 = self.conv1(x_even)
        x1 = self.dgcn(x1, adaptive_adj)
        d = x_odd.mul(torch.tanh(x1))

        x2 = self.conv2(x_odd)
        x2 = self.dgcn(x2, adaptive_adj)
        c = x_even.mul(torch.tanh(x2))

        x3 = self.conv3(c)
        x3 = self.dgcn(x3, adaptive_adj)
        x_odd_update = d + x3

        x4 = self.conv4(d)
        x4 = self.dgcn(x4, adaptive_adj)
        x_even_update = c + x4

        return (x_even_update, x_odd_update)


class IDGCN_Tree(nn.Module):
    def __init__(
        self, device, channels=64, diffusion_step=1, num_nodes=170, dropout=0.1
    ):
        super().__init__()

        self.IDGCN1 = IDGCN(
            device=device,
            splitting=True,
            channels=channels,
            diffusion_step=diffusion_step,
            num_nodes=num_nodes,
            dropout=dropout,
        )
        self.IDGCN2 = IDGCN(
            device=device,
            splitting=True,
            channels=channels,
            diffusion_step=diffusion_step,
            num_nodes=num_nodes,
            dropout=dropout,
        )
        self.IDGCN3 = IDGCN(
            device=device,
            splitting=True,
            channels=channels,
            diffusion_step=diffusion_step,
            num_nodes=num_nodes,
            dropout=dropout,
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
        x_even_update1, x_odd_update1 = self.IDGCN1(x)
        x_even_update2, x_odd_update2 = self.IDGCN2(x_even_update1)
        x_even_update3, x_odd_update3 = self.IDGCN3(x_odd_update1)
        concat1 = self.concat(x_even_update2, x_odd_update2)
        concat2 = self.concat(x_even_update3, x_odd_update3)
        concat0 = self.concat(concat1, concat2)
        output = concat0 + x
        return output


class STIDGCN(nn.Module):
    def __init__(
        self, device, input_dim, num_nodes, channels, granularity, dropout=0.1
    ):
        super().__init__()

        self.device = device
        self.num_nodes = num_nodes
        self.output_len = 12
        diffusion_step = 1

        self.Temb = TemporalEmbedding(granularity, channels)

        self.start_conv = nn.Conv2d(
            in_channels=input_dim, out_channels=channels, kernel_size=(1, 1)
        )

        self.tree = IDGCN_Tree(
            device=device,
            channels=channels,
            diffusion_step=diffusion_step,
            num_nodes=self.num_nodes,
            dropout=dropout,
        )

        self.glu = GLU(channels, dropout)

        self.regression_layer = nn.Conv2d(
            channels, self.output_len, kernel_size=(1, self.output_len)
        )

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, input):
        x = input
        # Encoder
        # Data Embedding
        time_emb = self.Temb(input.permute(0, 3, 2, 1))
        x = self.start_conv(x) + time_emb
        # IDGCN_Tree
        x = self.tree(x)
        # Decoder
        gcn = self.glu(x) + x
        prediction = self.regression_layer(F.relu(gcn))
        return prediction
