import itertools
import random
import sys

from sota.cnn.operations import *

sys.path.insert(0, '../../')
from nasbench201.utils import drop_path


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class GenCell(nn.Module):

    def __init__(self, arch, C_prev_prev, C_prev, C, reduction, reduction_prev, steps=None, concat=None, data1=None,
                 data2=None, primitives=None):
        super(GenCell, self).__init__()
        self.reduction = reduction
        self.primitives = primitives['primitives_reduct' if reduction else 'primitives_normal']
        self.multiplier = len(concat)
        self.cell_score = 0
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if arch:
            op_names, indices = zip(*arch)
            self.arch = arch
            self._compile(C, op_names, indices, concat, reduction)
        else:
            if (data1 is None) or (data2 is None) or (primitives is None) or (concat is None) or (steps is None):
                raise ValueError('Data or Primitives or concat or steps is None')
            else:
                self.topology = None
                self._concat = concat
                self._gen_network(C, data1, data2, steps)
                self.arch = [(self._ops_name[i], self._indices[i]) for i in range(len(self._ops))]

    def _gen_network(self, C, data1, data2, steps):
        data1 = self.preprocess0(data1)
        data2 = self.preprocess1(data2)
        self._ops = []
        states = [data1, data2]
        edge_index = 0
        self._steps = steps
        count = 0
        no_param_ops = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect']

        for i in range(self._steps):
            edges_in = []
            scores, maps, ops = [], [], []
            for j in range(2 + i):
                stride = 2 if self.reduction and j < 2 else 1
                for op_name in self.primitives[edge_index]:
                    op = OPS[op_name](C, stride, True)
                    # print(len(states))
                    map = op(states[j])
                    s = self.score(map)
                    maps.append(map)
                    ops.append((op, j, op_name))
                    scores.append(s)

            comb_scores = list(itertools.combinations(scores, 2))
            sorted_id = sorted(range(len(comb_scores)), key=lambda k: sum(comb_scores[k]), reverse=True)

            for id in sorted_id:
                ops_id = [scores.index(comb_scores[id][k]) for k in range(2)]
                prev_node_id = [ops[k][1] for k in ops_id]
                if len(set(prev_node_id)) == len(prev_node_id):
                    opnames = [ops[k][2] for k in ops_id]
                    c = sum([_ in no_param_ops for _ in opnames])
                    if (c < 2 and count + c < 3):
                        ops = [ops[_] for _ in ops_id]
                        edges_in = [maps[_] for _ in ops_id]
                        self.cell_score += sum(comb_scores[id])
                        count += c
                        break
                    else:
                        continue
                else:
                    continue
            out = sum(edges_in)
            self._ops += ops
            edge_index += 1
            states.append(out)
            _, self._indices, self._ops_name = zip(*self._ops)

        self._ops, self._indices, self._ops_name = zip(*self._ops)
        self._ops = nn.ModuleList(self._ops)


    def score(self, x):
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

    def get_arch(self):
        return self.arch

    def get_cell_score(self):
        return self.cell_score

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class GenCellV2(nn.Module):

    def __init__(self, arch, C_prev_prev, C_prev, C, reduction, reduction_prev, steps=None, concat=None, data1=None,
                 data2=None, primitives=None):
        super(GenCellV2, self).__init__()
        self.reduction = reduction
        self.primitives = primitives['primitives_reduct' if reduction else 'primitives_normal']
        self.multiplier = len(concat)
        self.cell_score = 0
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if arch:
            # print(arch)
            op_names, indices = zip(*arch)
            self.arch = arch
            self._compile(C, op_names, indices, concat, reduction, steps=steps)
        else:
            if (data1 is None) or (data2 is None) or (primitives is None) or (concat is None) or (steps is None):
                raise ValueError('Data or Primitives or concat or steps is None')
            else:
                self._concat = concat
                self._gen_network(C, data1, data2, steps)
                self.arch = [(self._ops_name[i], self._indices[i]) for i in range(len(self._ops))]

    def _gen_network(self, C, data1, data2, steps):
        data1 = self.preprocess0(data1)
        data2 = self.preprocess1(data2)
        data = data1 + data2
        self._ops = []
        states = [data]
        edge_index = 0
        self._steps = steps
        no_param_ops = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect']

        for i in range(self._steps):
            edges_in = []
            scores, maps, ops = [], [], []
            for j in range(i + 1):
                stride = 2 if self.reduction and j < 1 else 1
                for op_name in self.primitives[edge_index]:
                    op = OPS[op_name](C, stride, True)
                    map = op(states[j])
                    s = self.score(map)
                    maps.append(map)
                    ops.append((op, j, op_name))
                    scores.append(s)

            comb_scores = list(itertools.combinations(scores, i + 1))
            sorted_id = sorted(range(len(comb_scores)), key=lambda k: sum(comb_scores[k]), reverse=True)

            for id in sorted_id:
                ops_id = [scores.index(comb_scores[id][_]) for _ in range(i + 1)]
                prev_node_id = [ops[_][1] for _ in ops_id]
                if len(set(prev_node_id)) == len(prev_node_id):
                    opnames = [ops[_][2] for _ in ops_id]
                    c = sum([_ in no_param_ops for _ in opnames])
                    if c < 2:
                        ops = [ops[_] for _ in ops_id]
                        edges_in = [maps[_] for _ in ops_id]
                        self.cell_score += sum(comb_scores[id])
                        break
                    else:
                        continue
                else:
                    continue
            out = sum(edges_in)
            self._ops += ops
            edge_index += 1
            states.append(out)

        self._ops, self._indices, self._ops_name = zip(*self._ops)
        self._ops = nn.ModuleList(self._ops)

    def score(self, x):
        x = x[0]
        out_channels = x.size()[0]
        fea = x.reshape(out_channels, -1)
        temp = []
        for i in range(8):
            idxs = random.sample(range(out_channels), 8)
            corr = torch.corrcoef(fea[idxs, :])
            corr[torch.isnan(corr)] = 0
            corr[torch.isinf(corr)] = 0
            values = torch.linalg.eig(corr)[0]
            temp.append(torch.min(torch.real(values)))
        result = torch.mean(torch.tensor(temp)) * out_channels / 8
        return result

    def get_arch(self):
        return self.arch

    def get_cell_score(self):
        return self.cell_score

    def _compile(self, C, op_names, indices, concat, reduction, steps):
        assert len(op_names) == len(indices)
        self._steps = steps
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 1 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices


    def forward(self, s1, s2, drop_prob):
        s1 = self.preprocess0(s1)
        s2 = self.preprocess1(s2)
        s = s1 + s2
        states = [s]
        idx = 0
        for i in range(self._steps):
            edges_in = []
            for prev_node in range(i + 1):
                op = self._ops[idx]
                id = self._indices[idx]
                map = op(states[id])
                if self.training and drop_prob > 0.:
                    if not isinstance(op, Identity):
                        map = drop_path(map, drop_prob)
                edges_in.append(map)
                idx += 1
            s = sum(edges_in)
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHead(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHead, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            # image size = 2 x 2
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, drop_path_prob, genotype):
        super(Network, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        self.drop_path_prob = drop_path_prob

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class GenNetwork(nn.Module):

    def __init__(self, arch, C, num_classes, layers, auxiliary, steps, concat, data, primitives, drop_path_prob):
        super(GenNetwork, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = drop_path_prob
        self.net_score = 0
        if arch is None:
            arch = [None] * layers

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        s0 = s1 = self.stem(data)

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = GenCell(arch[i], C_prev_prev, C_prev, C_curr, reduction, reduction_prev, steps=steps, concat=concat,
                           data1=s0, data2=s1, primitives=primitives)
            self.net_score += cell.get_cell_score()
            s0, s1 = s1, cell(s0, s1, 0)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.arch = [cell.get_arch() for cell in self.cells]

    def get_arch(self):
        return self.arch

    def get_net_score(self):
        return self.net_score

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            # print(self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class GenNetworkV2(nn.Module):

    def __init__(self, arch, C, num_classes, layers, auxiliary, steps, concat, data, primitives, drop_path_prob):
        super(GenNetworkV2, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = drop_path_prob
        self.net_score = 0
        if arch is None:
            arch = [None] * layers

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        s0 = s1 = self.stem(data)

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = GenCellV2(arch[i], C_prev_prev, C_prev, C_curr, reduction, reduction_prev, steps=steps,
                             concat=concat,
                             data1=s0, data2=s1, primitives=primitives)
            self.net_score += cell.get_cell_score()
            s0, s1 = s1, cell(s0, s1, 0)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.arch = [cell.get_arch() for cell in self.cells]

    def get_arch(self):
        return self.arch

    def get_net_score(self):
        return self.net_score

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux
