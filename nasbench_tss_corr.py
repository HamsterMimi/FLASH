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

import argparse
import time
import random

import torch
import torch.optim as optim
from nats_bench import create
from torch.backends import cudnn
from tqdm import tqdm


from foresight.dataset import *
from foresight.models import nasbench2


def distill(result):
    result = result.split('\n')
    cifar10 = result[5].replace(' ', '').split(':')
    cifar100 = result[7].replace(' ', '').split(':')
    imagenet16 = result[9].replace(' ', '').split(':')

    cifar10_train = float(cifar10[1].strip(',test')[-7:-2].strip('='))
    cifar10_test = float(cifar10[2][-7:-2].strip('='))
    cifar100_train = float(cifar100[1].strip(',valid')[-7:-2].strip('='))
    cifar100_valid = float(cifar100[2].strip(',test')[-7:-2].strip('='))
    cifar100_test = float(cifar100[3][-7:-2].strip('='))
    imagenet16_train = float(imagenet16[1].strip(',valid')[-7:-2].strip('='))
    imagenet16_valid = float(imagenet16[2].strip(',test')[-7:-2].strip('='))
    imagenet16_test = float(imagenet16[3][-7:-2].strip('='))

    return cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
        cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test


def query(api, genotype):
    result = api.query_by_arch(genotype, hp='200')
    cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
        cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test = distill(result)
    print(f'cifar10 train {cifar10_train} test {cifar10_test}')
    print(f'cifar100 train {cifar100_train} valid {cifar100_valid} test {cifar100_test}')
    print(f'imagenet16 train {imagenet16_train} valid {imagenet16_valid} test {imagenet16_test}')
    return cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
        cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test


def tostr(ls):
    return '|' + ls[0][0] + '~' + str(ls[0][1]) + '|+|' + ls[1][0] + '~' + str(ls[1][1]) + '|' + ls[2][0] + '~' + str(
        ls[2][
            1]) + '|+|' + ls[3][0] + '~' + str(ls[3][1]) + '|' + ls[4][0] + '~' + str(ls[4][1]) + '|' + ls[5][
        0] + '~' + str(ls[5][
                           1]) + '|'

def tolist(str):
    strlist = str.split('+')
    ls = []
    for s in strlist:
        s = s.split('|')
        for i in s:
            if i != '':
                si = i.split('~')
                ls.append((si[0], int(si[1])))
    return ls






def get_num_classes(args):
    return 100 if args.dataset == 'cifar100' else 10 if args.dataset == 'cifar10' else 120


def setup_experiment(net, args):
    optimiser = optim.SGD(
        net.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=0, last_epoch=-1)
    cutout = -1 if 'ImageNet16' in args.dataset else args.cutout
    train_loader, val_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset,
                                                     args.num_data_workers, resize=None, auto_augment=args.autoaugment,
                                                     random_erase=args.erase, cutout=cutout)

    return optimiser, lr_scheduler, train_loader, val_loader

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
def parse_arguments():
    parser = argparse.ArgumentParser(description='EcoNAS Training Pipeline for NAS-Bench-201')
    parser.add_argument('--search_space', default='tss', type=str)
    parser.add_argument('--api_loc', default='data/NAS-Bench-201-v1_1-096897.pth',
                        type=str, help='path to API')
    parser.add_argument('--measure', default='meco', type=str, choices=['meco', 'param', 'flops', 'zen'])
    parser.add_argument('--learning_rate', default=0.025, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--outdir', default='./experiments',
                        type=str, help='output directory')
    parser.add_argument('--init_w_type', type=str, default='none',
                        help='weight initialization (before pruning) type [none, xavier, kaiming, zero, one]')
    parser.add_argument('--init_b_type', type=str, default='none',
                        help='bias initialization (before pruning) type [none, xavier, kaiming, zero, one]')

    parser.add_argument('--outfname', default='test',
                        type=str, help='output filename')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--init_channels', default=16, type=int)
    parser.add_argument('--version', default=2, type=int)
    parser.add_argument('--pool', default=10, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--cutout', default=-1, type=int)
    # parser.add_argument('--img_size', default=32, type=int)
    parser.add_argument('--autoaugment', default=False, type=bool)
    parser.add_argument('--erase', default=False, type=bool)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=0, type=int)

    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--num_data_workers', type=int, default=2, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
    parser.add_argument('--dataload_info', type=int, default=1,
                        help='number of batches to use for random dataload or number of samples per class for grasp dataload')
    parser.add_argument('--write_freq', type=int, default=5, help='frequency of write to file')
    parser.add_argument('--logmeasures', action="store_true", default=False,
                        help='add extra logging for predictive measures')
    parser.add_argument('--test', default=False, type=bool)
    parser.add_argument('--noacc', default=False, action='store_true',
                        help='avoid loading NASBench2 api an instead load a pickle file with tuple (index, arch_str)')
    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    return args


if __name__ == '__main__':
    args = parse_arguments()
    print(args.device)
    from nats_bench import create

    api = create(None, args.search_space, fast_mode=True, verbose=False)

    seed_torch(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




    pre = 'cf' if 'cifar' in args.dataset else 'im'
    op = f"{args.search_space}_{pre}{get_num_classes(args)}_{args.measure}.p"
    os.makedirs(op, exist_ok=True)
    op = os.path.join(args.outdir, op)


    train_loader, val_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset,
                                                     args.num_data_workers, resize=32, auto_augment=args.autoaugment,
                                                     random_erase=args.erase, cutout=-1)
    x = next(iter(train_loader))[0][0].unsqueeze(0)
    # arch = [('nor_conv_3x3', 0), ('nor_conv_3x3', 0), ('nor_conv_3x3', 1), ('nor_conv_3x3', 0), ('nor_conv_3x3', 1), ('nor_conv_3x3', 2)]
    # arch = None
    end = len(api) if args.end == 0 else args.end
    cached_res = []
    for i, arch_str in enumerate(api):
        if i < args.start:
            continue
        if i >= end:
            break

        arch = tolist(arch_str)
        net = nasbench2.gen_model(data=x, num_classes=get_num_classes(args),
                                    version=args.version, arch=arch,
                                    init_channels=args.init_channels, measure=args.measure)
        res = {'i': i, 'arch': arch_str}

        net_score = net.get_net_score()
        if not args.noacc:
            info = api.get_more_info(i, 'cifar10-valid' if args.dataset == 'cifar10' else args.dataset, iepoch=None,
                                    hp='200', is_random=False)

            trainacc = info['train-accuracy']
            valacc = info['valid-accuracy']
            testacc = info['test-accuracy']

            res['trainacc'] = trainacc
            res['valacc'] = valacc
            res['testacc'] = testacc

        res[f'{args.measure}'] = net_score
        print(res)
        cached_res.append(res)

        # write to file
        if i % args.write_freq == 0 or i == len(api) - 1 or i == 10:
            print(f'writing {len(cached_res)} results to {op}')
            pf = open(op, 'ab')
            for cr in cached_res:
                pickle.dump(cr, pf)
            pf.close()
            cached_res = []





