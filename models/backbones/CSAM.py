import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

class SandGlassBlock(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_c,
                                 out_features=in_c * 2,
                                 bias=False)
        self.bn1 = nn.BatchNorm1d(in_c * 2)
        self.linear2 = nn.Linear(in_features=in_c * 2,
                                 out_features=in_c,
                                 bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.linear1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.linear2(output)
        output = torch.tanh(output)
        output = 1 + output

        return output


class CSAM(nn.Module):

    def __init__(self, resnet, noise):

        super().__init__()

        # self.args = args
        self.resnet = resnet
        if resnet:
            self.in_c = 640
        else:
            self.in_c = 64

        self.prt_self = SandGlassBlock(self.in_c)
        self.prt_other = SandGlassBlock(self.in_c)
        self.qry_self = SandGlassBlock(self.in_c)
        self.noise = noise

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def add_noise(self, input):
        if self.training and self.noise:
            noise_value = 0.2
            # [15, 1, 640, 1]  [225, 640, 1]
            # noise = ((torch.rand(input.shape).to(input.device) - .5) * 2) * self.args.noise_value
            noise = ((torch.rand(input.shape).to(input.device) - .5) * 2) * noise_value
            input = input + noise  # [15, 1, 640, 1]  [225, 640, 1]
            input = input.clamp(min=0., max=2.)  # [15, 1, 640, 1]  [225, 640, 1]

        return input

    def dist(self, input, spt=False, normalize=True):

        if spt:
            way, c, m = input.shape
            input_C_gap = input.mean(dim=-2)

            input = input.reshape(way * c, m)
            input = input.unsqueeze(dim=1)
            input_C_gap = input_C_gap.unsqueeze(dim=0)

            dist = torch.sum(torch.pow(input - input_C_gap, 2), dim=-1)
            if normalize:
                dist = dist / m

            dist = dist.reshape(way, c, -1)
            dist = dist.transpose(-1, -2)

            indices_way = torch.arange(way)
            indices_1 = indices_way.repeat_interleave((way - 1))
            indices_2 = []
            for i in indices_way:
                indices_2_temp = torch.cat((indices_way[:i], indices_way[i + 1:]),
                                           dim=-1)
                indices_2.append(indices_2_temp)

            indices_2 = torch.cat(indices_2, dim=0)

            dist_self = dist[indices_way, indices_way]

            dist_other = dist[indices_1, indices_2]

            dist_other = dist_other.view(way, way-1, -1)


            return dist_self, dist_other

        else:
            batch, c, m = input.shape
            input_C_gap = input.mean(dim=-2).unsqueeze(dim=-2)

            dist = torch.sum(torch.pow(input - input_C_gap, 2), dim=-1)
            if normalize:
                dist = dist / m

            return dist

    def weight(self, spt, qry):

        way, shot, c, m = spt.shape
        batch, _, _, _ = qry.shape
        prt = spt.mean(dim=1)
        qry = qry.squeeze(dim=1)

        dist_prt_self, dist_prt_other = self.dist(prt, spt=True)
        dist_qry_self = self.dist(qry)

        dist_prt_self = dist_prt_self.view(-1, c)
        dist_prt_other, _ = dist_prt_other.min(dim=-2)
        dist_prt_other = dist_prt_other.view(-1, c)
        dist_qry_self = dist_qry_self.view(-1, c)

        weight_prt_self = self.prt_self(dist_prt_self)
        weight_prt_self = weight_prt_self.view(way, 1, c)
        weight_prt_other = self.prt_other(dist_prt_other)
        weight_prt_other = weight_prt_other.view(way, 1, c)
        weight_qry_self = self.qry_self(dist_qry_self)

        alpha_prt = 0.5
        alpha_prt_qry = 0.5

        beta_prt = 1. - alpha_prt
        beta_prt_qry = 1. - alpha_prt_qry

        weight_prt = alpha_prt * weight_prt_self + beta_prt * weight_prt_other  # [10, 1, 640]

        weight_prt = weight_prt.unsqueeze(-1)

        weight_qry_self = weight_qry_self.unsqueeze(-1)

        return weight_prt, weight_qry_self

    def forward(self, spt, qry):
        support = spt
        query = qry
        query = query.squeeze(1)

        weight_prt, weight_qry = self.weight(spt, qry)
        weight_prt = self.add_noise(weight_prt)
        weight_qry = self.add_noise(weight_qry)

        w_spt = support * weight_prt
        w_qry = query * weight_qry

        return w_spt, w_qry