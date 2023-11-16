# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import os
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from nasbench201.utils import drop_path




class ReLUConvBN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, affine, track_running_stats=True, use_bn=True, name='ReLUConvBN'):
        super(ReLUConvBN, self).__init__()
        self.name = name
        if use_bn:
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=not affine),
                nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=track_running_stats)
                )
        else:
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=not affine)
                )



    def forward(self, x):
        return self.op(x)

class Identity(nn.Module):
    def __init__(self, name='Identity'):
        self.name = name
        super(Identity, self).__init__()


    def forward(self, x):
        return x

class Zero(nn.Module):

  def __init__(self, stride, name='Zero'):
    self.name = name
    super(Zero, self).__init__()
    self.stride = stride


  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)

class POOLING(nn.Module):
    def __init__(self, kernel_size, stride, padding, name='POOLING'):
        super(POOLING, self).__init__()
        self.name = name
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=1, count_include_pad=False)

    def forward(self, x):
        return self.avgpool(x)


class reduction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(reduction, self).__init__()
        self.residual = nn.Sequential(
                            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False))

        self.conv_a = ReLUConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, dilation=1, affine=True, track_running_stats=True)
        self.conv_b = ReLUConvBN(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1, affine=True, track_running_stats=True)

    def forward(self, x):
        basicblock = self.conv_a(x)
        basicblock = self.conv_b(basicblock)
        residual = self.residual(x)
        return residual + basicblock

class stem(nn.Module):
    def __init__(self, out_channels, use_bn=True):
        super(stem, self).__init__()
        if use_bn:
            self.net = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels))
        else:
            self.net = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
            )

    def forward(self, x):
        return self.net(x)

class top(nn.Module):
    # def __init__(self, in_dims, num_classes, use_bn=True):
    def __init__(self, in_dims, use_bn=True):
        super(top, self).__init__()
        if use_bn:
            self.lastact = nn.Sequential(nn.BatchNorm2d(in_dims), nn.ReLU(inplace=True))
        else:
            self.lastact = nn.ReLU(inplace=True)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Linear(in_dims, num_classes)

    def forward(self, x):
        x = self.lastact(x)
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        # logits = self.classifier(x)
        # return logits
        return x


class SearchCell(nn.Module):

    def __init__(self, in_channels, out_channels, stride, affine, track_running_stats, use_bn=True, num_nodes=4, keep_mask=None):
        super(SearchCell, self).__init__()
        self.num_nodes = num_nodes
        self.options = nn.ModuleList()
        for curr_node in range(self.num_nodes-1):
            for prev_node in range(curr_node+1): 
                for _op_name in OPS.keys():
                    op = OPS[_op_name](in_channels, out_channels, stride, affine, track_running_stats, use_bn)
                    self.options.append(op)

        if keep_mask is not None:
            self.keep_mask = keep_mask
        else:
            self.keep_mask = [True]*len(self.options)

    def forward(self, x):
        outs = [x]

        idx = 0
        # print(self.keep_mask)
        for curr_node in range(self.num_nodes-1):
            edges_in = []
            for prev_node in range(curr_node+1): # n-1 prev nodes
                for op_idx in range(len(OPS.keys())):
                    if self.keep_mask[idx]:
                        # print(list(OPS.keys())[op_idx])
                        edges_in.append(self.options[idx](outs[prev_node]))
                        # print(f'({op_idx}, {idx}),({prev_node}, {curr_node + 1})', self.options[idx].name)
                    idx += 1
            node_output = sum(edges_in)
            outs.append(node_output)
        return outs[-1]


class GenCell(nn.Module):

    def __init__(self, data, in_channels, out_channels, stride, affine, track_running_stats, use_bn=True, num_nodes=4, arch=None):
        super(GenCell, self).__init__()
        self.num_nodes = num_nodes
        self.options = nn.ModuleList()
        self.indicate = []
        self.cell_score = 0

        if arch == None:
            self.arch = []

            self._gen_network(data, in_channels, out_channels, stride, affine, track_running_stats, use_bn)
        else:
            self.arch = arch

            for op_name in arch:
                # e.g. ['nor_conv_3x3', 'nor_conv_3x3', 'nor_conv_3x3', 'skip_connect', 'nor_conv_3x3', 'nor_conv_1x1']
                self.options.append(OPS[op_name[0]](in_channels, out_channels, stride, affine, track_running_stats, use_bn))
                self.indicate.append(op_name[1])




    def _gen_network(self, data, in_channels, out_channels, stride, affine, track_running_stats, use_bn):
        # print('gen_network')
        outs = [data]
        count = 0
        for curr_node in range(self.num_nodes - 1):
            edges_in = []
            scores, ops, maps = [], [], []
            for prev_node in range(curr_node + 1):
                for _op_name in OPS.keys():
                    op = OPS[_op_name](in_channels, out_channels, stride, affine, track_running_stats, use_bn)
                    # print(_op_name, outs[prev_node].size())
                    map = op(outs[prev_node])
                    meco = self.meco(map)
                    scores.append(meco)
                    maps.append(map)
                    ops.append((op, prev_node, _op_name))

            # sorted_id = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
            no_param_ops = ['avg_pool_3x3', 'skip_connect', 'none']

            comb_scores = list(itertools.combinations(scores, curr_node + 1))
            # comb_maps = list(itertools.combinations(maps, curr_node + 1))
            sorted_id = sorted(range(len(comb_scores)), key=lambda k: sum(comb_scores[k]), reverse=True)
            # sorted_map = sorted(range(len(comb_scores)), key=lambda k: self.meco(sum(comb_maps[k])), reverse=True)
            # print(sorted_map)
            # for id in sorted_id:
            #     print(comb_scores[id], sum(comb_scores[id]))
            # for id in sorted_map:
            #     ops_id = [self._index(maps, comb_maps[id][_]) for _ in range(curr_node + 1)]
            #     # print(ops_id)
            #     prev_node_id = [ops[_][1] for _ in ops_id]
            #     if len(set(prev_node_id)) == len(prev_node_id):
            #         # print(prev_node_id)
            #         opnames = [ops[_][2] for _ in ops_id]
            #         c = sum([_ in no_param_ops for _ in opnames])
            #         # print(opnames, c)
            #         if c < 2:
            #             ops = [ops[_] for _ in ops_id]
            #             edges_in = [maps[_] for _ in ops_id]
            #             count += c
            #             # print([(_[2],_[1]) for _ in ops], count)
            #             break
            #         else:
            #             continue
            #     else:
            #         continue
            # out = sum(edges_in)
            # self.cell_score += self.meco(out)
            # outs.append(out)
            # for o in ops:
            #     self.options.append(o[0])
            #     self.indicate.append(o[1])
            #     self.arch.append((o[2], o[1]))
        # print(self.cell_score)

            for id in sorted_id:
                ops_id = [scores.index(comb_scores[id][_]) for _ in range(curr_node + 1)]
                prev_node_id = [ops[_][1] for _ in ops_id]
                if len(set(prev_node_id)) == len(prev_node_id):
                    # print(prev_node_id)
                    opnames = [ops[_][2] for _ in ops_id]
                    c = sum([_ in no_param_ops for _ in opnames])
                    # print(opnames, c)
                    if c < 2:
                        ops = [ops[_] for _ in ops_id]
                        edges_in = [maps[_] for _ in ops_id]
                        self.cell_score += sum(comb_scores[id]).item()
                        count += c
                        # print([(_[2],_[1]) for _ in ops], count)
                        break
                    else:
                        continue
                else:
                    continue
            out = sum(edges_in)
            outs.append(out)
            for o in ops:
                self.options.append(o[0])
                self.indicate.append(o[1])
                self.arch.append((o[2], o[1]))
            # print(self.arch)
            #
            # print(len(self.options))
            # print(self.options)
            # print([(o[2], o[1]) for o in ops])


    def forward(self, x, drop_prob):
        outs = [x]
        idx = 0
        for curr_node in range(self.num_nodes - 1):
            edges_in = []
            for prev_node in range(curr_node + 1):  # n-1 prev nodes
                op = self.options[idx]
                id = self.indicate[idx]
                map = op(outs[id])
                if self.training and drop_prob > 0.:
                    if not isinstance(op, Identity):
                        map = drop_path(map, drop_prob)
                edges_in.append(map)
                idx += 1
            node_output = sum(edges_in)
            outs.append(node_output)
        return outs[-1]

    def _index(self, l, t):
        for ele in range(len(l)):
            if l[ele].equal(t):
                return ele

    def get_arch(self):
        return self.arch

    def get_cell_score(self):
        return self.cell_score

    def meco(self, x):
        x = x[0]
        out_channels = x.size()[0]
        fea = x.reshape(out_channels, -1)
        temp = []
        for i in range(int(out_channels/8)):
            idxs = random.sample(range(out_channels), 8)
            corr = torch.corrcoef(fea[idxs, :])
            corr[torch.isnan(corr)] = 0
            corr[torch.isinf(corr)] = 0
            values = torch.linalg.eig(corr)[0]
            temp.append(torch.min(torch.real(values)))
        result = torch.mean(torch.tensor(temp))
        return result

    def deco(self, x, out_channels):
        x = x[0]
        fea = x.reshape(out_channels, -1)
        temp = []
        for i in range(8):
            idxs = random.sample(range(out_channels), 8)
            corr = torch.corrcoef(fea[idxs, :])
            corr[torch.isnan(corr)] = 0
            corr[torch.isinf(corr)] = 0
            values = torch.linalg.eig(corr)[0]
            temp.append(torch.min(torch.real(values))/torch.max(torch.real(values)))
            # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.mean(torch.tensor(temp))
        return result


OPS = {
    'none' : lambda in_channels, out_channels, stride, affine, track_running_stats, use_bn: Zero(stride, name='none'),
    'avg_pool_3x3' : lambda in_channels, out_channels, stride, affine, track_running_stats, use_bn: POOLING(3, 1, 1, name='avg_pool_3x3'),
    'nor_conv_3x3' : lambda in_channels, out_channels, stride, affine, track_running_stats, use_bn: ReLUConvBN(in_channels, out_channels, 3, 1, 1, 1, affine, track_running_stats, use_bn, name='nor_conv_3x3'),
    'nor_conv_1x1' : lambda in_channels, out_channels, stride, affine, track_running_stats, use_bn: ReLUConvBN(in_channels, out_channels, 1, 1, 0, 1, affine, track_running_stats, use_bn, name='nor_conv_1x1'),
    'skip_connect' : lambda in_channels, out_channels, stride, affine, track_running_stats, use_bn: Identity(name='skip_connect'),
}

if __name__ == '__main__':
    s = stem(out_channels=16)
    data = torch.randn(size=(1, 3, 32, 32))
    data = s(data)
    net = GenCell(data, in_channels=16, out_channels=16, stride=1, affine=False, track_running_stats=False,
                       use_bn=True, arch=['nor_conv_3x3','nor_conv_3x3','nor_conv_3x3','skip_connect','nor_conv_3x3','nor_conv_1x1'])
    print(net)
    print(net.get_arch())
