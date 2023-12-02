import os
import random

import numpy as np
import torch.nn as nn
import math
import sys

from torch.backends import cudnn
from tqdm import tqdm

sys.path.insert(0, '../../')
from mobilenet_search_space.retrain_architecture.torch_blocks import *


def get_net_score(net, x):
    result_list = []

    def forward_hook(module, data_input, data_output):
        try:
            fea = data_output[0].clone().detach()
            n = torch.tensor(fea.shape[0])
            fea = fea.reshape(n, -1)
            idxs = random.sample(range(n), 8)
            corr = torch.corrcoef(fea[idxs, :])
            corr[torch.isnan(corr)] = 0
            corr[torch.isinf(corr)] = 0
            values = torch.linalg.eig(corr)[0]
            result = torch.min(torch.real(values))
            result_list.append(result * (n / 8))
        except:
            result_list.append(0)

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)
    x = x.cuda()
    net = net.cuda()
    net(x)
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    results = results[torch.logical_not(torch.isinf(results))]
    res = torch.sum(results)
    result_list.clear()
    # feature_list.clear()
    res = res.item()
    return res

def get_feature_score(x):
    feature = x[0]
    n = torch.tensor(feature.shape[0])
    feature = feature.reshape(n, -1)
    idxs = random.sample(range(n), 8)
    corr = torch.corrcoef(feature[idxs, :])
    corr[torch.isnan(corr)] = 0
    corr[torch.isinf(corr)] = 0
    values = torch.linalg.eig(corr)[0]
    result = torch.min(torch.real(values))
    return result * (n / 8)



def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class Select_one_OP(nn.Module):
  def __init__(self, inp, oup, stride):
    super(Select_one_OP, self).__init__()
    self._ops = nn.ModuleList()
    self.input_channel = inp
    self.output_channel = oup
    self.stride = stride
    for idx, key in enumerate(config.blocks_keys):
      op = blocks_dict[key](inp, oup, stride)
      op.idx = idx
      self._ops.append(op)

  def forward(self, x, id):
    # if id < 0:
    #     return x
    return self._ops[id](x)


class Network(nn.Module):
    def __init__(self, rngs, n_class=1000, input_size=224, width_mult=1.):
        super(Network, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [  #for GPU search
            # t, c, n, s
            [6, 32,  4, 2],
            [6, 56,  4, 2],
            [6, 112, 4, 2],
            [6, 128, 4, 1],
            [6, 256, 4, 2],
            [6, 432, 1, 1],
        ]
        # building first layer
        input_channel = int(40 * width_mult)
        self.last_channel = int(1728 * width_mult) if width_mult > 1.0 else 1728
        self.conv_bn = conv_bn(3, input_channel, 2)
        self.MBConv_ratio_1 = InvertedResidual(input_channel, int(24*width_mult), 3, 1, 1, 1)
        input_channel = int(24*width_mult)
        self.features = []
        num = 0
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                rng = rngs[num]
                num += 1
                if rng < 0:
                    continue
                if i == 0:
                    op = blocks_dict[blocks_keys[rng]](input_channel, output_channel, s)
                    self.features.append(op)
                else:
                    op = blocks_dict[blocks_keys[rng]](input_channel, output_channel, 1)
                    self.features.append(op)
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)
        # building last several layers
        self.conv_1x1_bn = conv_1x1_bn(input_channel, self.last_channel)
        self.avgpool = nn.AvgPool2d(input_size//32)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

    def forward(self, x, rngs=None):
        x = self.conv_bn(x)
        x = self.MBConv_ratio_1(x)
        x = self.features(x)
        x = self.conv_1x1_bn(x)
        x = self.avgpool(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def architecture(self):
        arch = []
        for feat in self.features:
            if feat.stride == 2:
                arch.append('{}(reduce, oup={})'.format(feat.type, feat.oup))
            else:
                arch.append('{}(normal, oup={})'.format(feat.type, feat.oup))
        return arch


class GzNetwork(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(GzNetwork, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [  #for GPU search
            # t, c, n, s
            [6, 32,  4, 2],
            [6, 56,  4, 2],
            [6, 112, 4, 2],
            [6, 128, 4, 1],
            [6, 256, 4, 2],
            [6, 432, 1, 1],
        ]
        # building first layer
        input_channel = int(40 * width_mult)
        self.last_channel = int(1728 * width_mult) if width_mult > 1.0 else 1728
        self.conv_bn = conv_bn(3, input_channel, 2)
        self.MBConv_ratio_1 = InvertedResidual(input_channel, int(24*width_mult), 3, 1, 1, 1)
        input_channel = int(24*width_mult)
        self.features = [] # 最终操作
        self.score = 0
        self.rng = []
        num = 0
        self.x = torch.randn(size=(1, 3, 224, 224))
        self.x = self.conv_bn(self.x)
        self.x = self.MBConv_ratio_1(self.x)
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                num += 1
                max_score = 0
                best_op_idx = -1
                best_op = None
                for i in range(-1, 6):
                    if i == -1:
                        score = 0
                        op_idx = -1
                        if score > max_score:
                            max_score = score
                            best_op_idx = op_idx
                            best_op = op
                    elif i == 0:
                        op = blocks_dict[blocks_keys[i]](input_channel, output_channel, s)
                        score = get_feature_score(op(self.x))
                        op_idx = 0
                        if score > max_score:
                            max_score = score
                            best_op_idx = op_idx
                            best_op = op
                    else:
                        op = blocks_dict[blocks_keys[i]](input_channel, output_channel, 1)
                        score = get_feature_score(op(self.x))
                        op_idx = i
                        if score > max_score:
                            max_score = score
                            best_op_idx = op_idx
                            best_op = op
                self.rng.append(best_op_idx)
                self.features.append(best_op)
                self.x = self.x if best_op is None else best_op(self.x)
                self.score += max_score
                input_channel = input_channel if best_op is None else output_channel

    def arch(self):
        return self.rng

    def get_score(self):
        return self.score



    def architecture(self):
        arch = []
        for feat in self.features:
            if feat.stride == 2:
                arch.append('{}(reduce, oup={})'.format(feat.type, feat.oup))
            else:
                arch.append('{}(normal, oup={})'.format(feat.type, feat.oup))
        return arch

if __name__ == '__main__':
    # arch = '5 2 2 4 2 5 3 4 1 1 5 3 5 1 3 1 1 5 1 5 3'
    # genotype_list = arch.split(' ')
    # genotype_list = [int(_) for _ in genotype_list]
    # model = Network(genotype_list)
    # x = torch.randn(size=(1, 3, 224, 224))
    # get_score(model, x)
    def seed_torch(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    seed_torch(42)
    best_network = None
    best_score = 0
    import time
    start = time.time()
    for i in tqdm(range(10)):

        net = GzNetwork()
        if net.get_score() > best_score:
            best_network = net.arch()
            best_score = net.get_score()
            # print(net.arch())
    end = time.time()
    print(f'best: {best_network}, time: {end-start}')

