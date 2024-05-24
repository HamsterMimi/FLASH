import itertools
import random
import sys

from thop import profile

sys.path.insert(0, '../../')
# from optimizers.darts.operations import *
from sota.cnn.operations import *


# from optimizers.darts.utils import drop_path

def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype)
            concat = range(2, 6)
        else:
            op_names, indices = zip(*genotype)
            concat = range(2, 6)
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

    def __init__(self, C_prev_prev, C_prev, C, reduction, reduction_prev, data1, data2, arch=None, primitives=None, measure='meco'):
        super(GenCell, self).__init__()
        print(C_prev_prev, C_prev, C)
        self.data1 = data1
        self.data2 = data2
        self.measure = measure
        self.reduction = reduction
        self.reduction_prev = reduction_prev
        self.cell_score = 0
        self.multiplier=4
        self._concat = range(2, 6)

        self.primitives = primitives['primitives_reduct' if reduction else 'primitives_normal']

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if arch:
            op_names, indices = zip(*arch)
            self.arch = arch
            self._compile(C, op_names, indices, self._concat, reduction)
        else:
            steps = 4
            if (data1 is None) or (data2 is None):
                raise ValueError('Data or Primitives or concat or steps is None')
            else:
                self.topology = None

                self._gen_network(C, data1, data2, steps)
                self.arch = [(self._ops_name[i], self._indices[i]) for i in range(len(self._ops))]


    def score(self, x, op=None, measure='meco', input_data=None):
        if measure == 'meco':
            x = x[0]
            out_channels = x.size()[0]
            fea = x.reshape(out_channels, -1)
            temp = []
            for i in range(int(out_channels / 8)):
                idxs = random.sample(range(out_channels), 8)
                corr = torch.corrcoef(fea[idxs, :])
                corr[torch.isnan(corr)] = 0
                corr[torch.isinf(corr)] = 0
                values = torch.linalg.eig(corr)[0]
                temp.append(torch.min(torch.real(values)) * out_channels / 8)
            result = torch.mean(torch.tensor(temp))
            return result
        elif measure == 'param' or measure == 'flops':
            flops, params = profile(op, inputs=(input_data,), verbose=False)
            if measure == 'param':
                return params
            elif measure == 'flops':
                return flops
        elif measure == 'zen':
            input = input_data
            input2 = torch.randn_like(input_data)
            mixup_input = input + 1e-2 * input2
            output = op(input)
            mixup_output = op(mixup_input)
            nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
            nas_score = torch.mean(nas_score)
            log_bn_scaling_factor = 0.0
            for m in op.modules():
                if isinstance(m, nn.BatchNorm2d):
                    try:
                        bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                        log_bn_scaling_factor += torch.log(bn_scaling_factor)
                    except:
                        pass
                pass
            pass
            nas_score = torch.log(nas_score) + log_bn_scaling_factor
            return nas_score
        else:
            raise NotImplementedError('Unknown measure')

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
                    s = self.score(map, op, self.measure, input_data=states[j])
                    if (self.measure == 'param' or self.measure == 'flops'):
                        t = j
                    else:
                        t = 0
                    s += t
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


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
            # Commenting it out for consistency with the experiments in the paper.
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkImageNet(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = 0.0

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype[i], C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux

class GenNetworkImageNet(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, x,  primitives, arch=None, drop_path_prob=0.0, measure='meco'):
        super(GenNetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = drop_path_prob
        self.x = x
        self.score = 0
        if arch is None:
            arch = [None] * layers
        # print(arch)
        self.measure = measure


        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )
        self.x0 = self.stem0(self.x)
        self.x1 = self.stem1(self.x0)

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = GenCell(C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.x0, self.x1,primitives=primitives, arch=arch[i],measure=self.measure)
            self.x0, self.x1 = self.x1, cell(self.x0, self.x1, drop_prob=0.)
            self.score += cell.get_cell_score()
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.arch = [cell.get_arch() for cell in self.cells]

    def get_arch(self):
        return self.arch
    def get_net_score(self):
        return self.score

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux
