import torch
import torch.nn as nn
from torch.nn import Conv2d, Parameter
import torch.nn.functional as F


def nconv(x, A):
    return torch.einsum('bcnt,nm->bcmt', (x, A)).contiguous()


class Diffusion_GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=2, order=1):
        super().__init__()
        c_in = (order * support_len + 1) * c_in
        self.conv = Conv2d(c_in, c_out, (1, 1), padding=(
            0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support: list):
        out = [x]
        for a in support:
            x1 = nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.conv(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
    
    
def sample_gumbel(device, shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(device, logits, temperature,  eps=1e-10):
    sample = sample_gumbel(device, logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(device, logits, temperature, hard=False, eps=1e-10):
    y_soft = gumbel_softmax_sample(
        device, logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).to(device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


class Graph_Generator(nn.Module):
    def __init__(self, device, channels, num_nodes, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.node = c_out
        self.device = device
        self.fc0 = nn.Linear(channels, num_nodes)
        self.fc1 = nn.Linear(num_nodes, 2*num_nodes)
        self.fc2 = nn.Linear(2*num_nodes, num_nodes)
        self.diffusion_conv = Diffusion_GCN(channels, channels, dropout, support_len=1)

    def forward(self, x, adj):
        x = self.diffusion_conv(x, [adj])
        x = x.sum(0)
        x = x.sum(2)
        x = x.permute(1, 0)
        x = self.fc0(x)
        x = torch.tanh(x)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.log(F.softmax(x, dim=-1))
        x = gumbel_softmax(self.device, x, temperature=0.5, hard=True)
        mask = torch.eye(x.shape[0], x.shape[0]).bool().to(device=self.device)
        x.masked_fill_(mask, 0)
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
    def __init__(self, device, channels, splitting=True, num_nodes=170, dropout=0.25, pre_adj=None, pre_adj_len=1):
        super(IDGCN, self).__init__()

        device = device
        self.dropout = dropout
        self.pre_adj_len = pre_adj_len
        self.num_nodes = num_nodes
        self.splitting = splitting
        self.pre_graph = pre_adj or []
        self.split = Splitting()

        Conv1 = []
        Conv2 = []
        Conv3 = []
        Conv4 = []
        pad_l = 3
        pad_r = 3

        apt_size = 10
        aptinit = pre_adj[0]
        self.pre_adj_len = 1
        nodevecs = self.svd_init(apt_size, aptinit)
        self.nodevec1, self.nodevec2 = [
            Parameter(n.to(device), requires_grad=True) for n in nodevecs]

        Conv1 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 5), dilation=1, stride=1, groups=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 3), dilation=1, stride=1, groups=1),
            nn.Tanh(),
        ]
        Conv2 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 5), dilation=1, stride=1, groups=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 3), dilation=1, stride=1, groups=1),
            nn.Tanh()
        ]

        Conv4 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 5), dilation=1, stride=1, groups=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 3), dilation=1, stride=1, groups=1),
            nn.Tanh()
        ]
        Conv3 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 5), dilation=1, stride=1, groups=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 3), dilation=1, stride=1, groups=1),
            nn.Tanh()
        ]

        self.conv1 = nn.Sequential(*Conv1)
        self.conv2 = nn.Sequential(*Conv2)
        self.conv3 = nn.Sequential(*Conv3)
        self.conv4 = nn.Sequential(*Conv4)

        self.a = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)

        self.graph_generator = Graph_Generator(
            device, channels, num_nodes)

        self.diffusion_conv = Diffusion_GCN(
            channels, channels, dropout, support_len=self.pre_adj_len)

    @staticmethod
    def svd_init(apt_size, aptinit):
        m, p, n = torch.svd(aptinit)
        nodevec1 = torch.mm(m[:, :apt_size], torch.diag(p[:apt_size] ** 0.5))
        nodevec2 = torch.mm(torch.diag(
            p[:apt_size] ** 0.5), n[:, :apt_size].t())
        return nodevec1, nodevec2

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        adaptive_adj = F.softmax(
            F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        #             x
        # xeven               xodd
        #         x1      x2
        #     c      dgcn     d
        #         x3      x4
        # xevenup             xoddup

        x1 = self.conv1(x_even)
        learn_adj = self.graph_generator(x1, adaptive_adj)
        dadj = learn_adj*self.a+adaptive_adj*(1-self.a)
        x1 = x1+self.diffusion_conv(x1, [dadj])
        d = x_odd.mul(torch.tanh(x1))

        x2 = self.conv2(x_odd)
        learn_adj = self.graph_generator(x2, adaptive_adj)
        dadj = learn_adj*self.a+adaptive_adj*(1-self.a)
        x2 = x2+self.diffusion_conv(x2, [dadj])
        c = x_even.mul(torch.tanh(x2))

        x3 = self.conv3(c)
        learn_adj = self.graph_generator(x3, adaptive_adj)
        dadj = learn_adj*self.a+adaptive_adj*(1-self.a)
        x3 = x3+self.diffusion_conv(x3, [dadj])
        x_odd_update = d - x3  #

        x4 = self.conv4(d)
        learn_adj = self.graph_generator(x4, adaptive_adj)
        dadj = learn_adj*self.a+adaptive_adj*(1-self.a)
        x4 = x4+self.diffusion_conv(x4, [dadj])
        x_even_update = c + x4

        return (x_even_update, x_odd_update, dadj)


class IDGCN_Tree(nn.Module):
    def __init__(self, device, num_nodes, channels, num_levels, dropout, pre_adj=None, pre_adj_len=1):
        super().__init__()
        self.levels = num_levels
        self.pre_graph = pre_adj or []

        self.IDGCN1 = IDGCN(splitting=True, channels=channels, device=device,
                            pre_adj=pre_adj, num_nodes=num_nodes, dropout=dropout, pre_adj_len=pre_adj_len)
        self.IDGCN2 = IDGCN(splitting=True, channels=channels, device=device,
                            pre_adj=pre_adj, num_nodes=num_nodes, dropout=dropout, pre_adj_len=pre_adj_len)
        self.IDGCN3 = IDGCN(splitting=True, channels=channels, device=device,
                            pre_adj=pre_adj, num_nodes=num_nodes, dropout=dropout, pre_adj_len=pre_adj_len)

        self.a = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)
        self.b = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)
        self.c = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)

    def concat(self, even, odd):
        even = even.permute(3, 1, 2, 0)
        odd = odd.permute(3, 1, 2, 0)
        len = even.shape[0]
        _ = []
        for i in range(len):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        return torch.cat(_, 0).permute(3, 1, 2, 0)

    def forward(self, x, adj):
        x_even_update1, x_odd_update1, dadj1 = self.IDGCN1(x)
        x_even_update2, x_odd_update2, dadj2 = self.IDGCN2(x_even_update1)
        x_even_update3, x_odd_update3, dadj3 = self.IDGCN3(x_odd_update1)
        concat1 = self.concat(x_even_update2, x_odd_update2)
        concat2 = self.concat(x_even_update3, x_odd_update3)
        concat0 = self.concat(concat1, concat2)
        adj = dadj1*self.a+dadj2*self.b+dadj3*self.c
        return concat0, adj


class STIDGCN(nn.Module):
    def __init__(self, device, num_nodes, channels, dropout=0.25, pre_adj=None):
        super().__init__()

        self.dropout = dropout
        self.num_nodes = num_nodes
        self.input_len = 12
        self.output_len = 12
        self.num_levels = 2
        self.groups = 1
        input_channel = 1
        apt_size = 10

        self.pre_graph = pre_adj or []
        self.pre_adj_len = len(self.pre_graph)+1

        aptinit = pre_adj[0]
        nodevecs = self.svd_init(apt_size, aptinit)
        self.nodevec1, self.nodevec2 = [
            Parameter(n.to(device), requires_grad=True) for n in nodevecs]

        self.a = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)

        self.start_conv = nn.Conv2d(in_channels=input_channel,
                                    out_channels=channels,
                                    kernel_size=(1, 1))

        self.tree = IDGCN_Tree(
            device=device,
            channels=channels,
            num_nodes=self.num_nodes,
            num_levels=self.num_levels,
            dropout=self.dropout,
            pre_adj_len=self.pre_adj_len,
            pre_adj=pre_adj
        )

        self.diffusion_conv = Diffusion_GCN(
            channels, channels, dropout, support_len=self.pre_adj_len)

        self.Conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=256,
                               kernel_size=(1, 1), stride=(1, 1))
        self.Conv2 = nn.Conv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=(1, 12), stride=(1, 1))
        self.Conv3 = nn.Conv2d(in_channels=512,
                               out_channels=12,
                               kernel_size=(1, 1), stride=(1, 1))

    @staticmethod
    def svd_init(apt_size, aptinit):
        m, p, n = torch.svd(aptinit)
        nodevec1 = torch.mm(m[:, :apt_size], torch.diag(p[:apt_size] ** 0.5))
        nodevec2 = torch.mm(torch.diag(
            p[:apt_size] ** 0.5), n[:, :apt_size].t())
        return nodevec1, nodevec2

    def forward(self, input):
        x = input

        x = self.start_conv(x)

        adaptive_adj = F.softmax(
            F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        skip = x
        x, dadj = self.tree(x, adaptive_adj)
        x = skip + x

        adj = self.a*adaptive_adj+(1-self.a)*dadj
        adj = self.pre_graph + [adj]

        gcn = self.diffusion_conv(x, adj)

        x = gcn + x

        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = self.Conv3(x)

        return x
