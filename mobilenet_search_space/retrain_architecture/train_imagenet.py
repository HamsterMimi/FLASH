import argparse
import glob
import json
import logging
import sys

import PIL
import torchvision
from torch.utils.tensorboard import SummaryWriter
import PIL
from thop import profile

# from genotypes import Genotype
import hotfix
from model import Network

sys.path.insert(0, '../../')
import time
import random
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.transforms as transforms
import nasbench201.utils as utils
import torch.nn.functional as F


parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='/dev/ImageNet-1K', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--sample_size', default=8, type=int)
parser.add_argument('--sample_count', default=1, type=int)
parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
parser.add_argument('--warmup', default=5, type=int)
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--label_smoothing', default=True, type=bool)
parser.add_argument('--mixup', default=False, type=bool)
parser.add_argument('--arch', default='gz_mobile_net', type=str)
parser.add_argument('--epochs', type=int, default=480, help='num of training epochs')
parser.add_argument('--nesterov', type=bool, default=True)
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random_ws seed')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--load', action='store_true', help='whether load checkpoint for continue training')
parser.add_argument('--lr_mode', default='cosine', type=str)
parser.add_argument('--lr_stage_list', default=None, type=str, help='stage-wise learning epoch list.')
parser.add_argument('--lr_stage_decay', default=None, type=float, help='stage-wise learning epoch list.')
# BN layer
parser.add_argument('--bn_momentum', type=float, default=0.01)
parser.add_argument('--bn_eps', type=float, default=None)

parser.add_argument('--parallel', default=True)

args = parser.parse_args()
args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

args.save = '../../experiments/sota/imagenet/eval/{}-{}-{}-{}'.format(
    args.save, time.strftime("%Y%m%d-%H%M%S"), args.arch, args.seed)
args.save += '-' + str(np.random.randint(10000))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(args.save + '/runs')

CLASSES = 1000

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def cross_entropy(logit, target):
    # target must be one-hot format!!
    prob_logit = F.log_softmax(logit, dim=1)
    loss = -(target * prob_logit).sum(dim=1).mean()
    return loss


class LearningRateScheduler():
    def __init__(self,
                 mode,
                 lr,
                 target_lr=None,
                 num_training_instances=None,
                 stop_epoch=None,
                 warmup_epoch=None,
                 stage_list=None,
                 stage_decay=None,
                 ):
        self.mode = mode
        self.lr = lr
        self.target_lr = target_lr if target_lr is not None else 0
        self.num_training_instances = num_training_instances if num_training_instances is not None else 1
        self.stop_epoch = stop_epoch if stop_epoch is not None else np.inf
        self.warmup_epoch = warmup_epoch if warmup_epoch is not None else 0
        self.stage_list = stage_list if stage_list is not None else None
        self.stage_decay = stage_decay if stage_decay is not None else 0

        self.num_received_training_instances = 0

        if self.stage_list is not None:
            self.stage_list = [int(x) for x in self.stage_list.split(',')]

    def update_lr(self, batch_size):
        self.num_received_training_instances += batch_size

    def get_lr(self, num_received_training_instances=None):
        if num_received_training_instances is None:
            num_received_training_instances = self.num_received_training_instances

        # start_instances = self.num_training_instances * self.start_epoch
        stop_instances = self.num_training_instances * self.stop_epoch
        warmup_instances = self.num_training_instances * self.warmup_epoch

        assert stop_instances > warmup_instances

        current_epoch = self.num_received_training_instances // self.num_training_instances

        if num_received_training_instances < warmup_instances:
            return float(num_received_training_instances + 1) / float(warmup_instances) * self.lr

        ratio_epoch = float(num_received_training_instances - warmup_instances + 1) / \
                      float(stop_instances - warmup_instances)

        if self.mode == 'cosine':
            factor = (1 - np.math.cos(np.math.pi * ratio_epoch)) / 2.0
            return self.lr + (self.target_lr - self.lr) * factor
        elif self.mode == 'stagedecay':
            stage_lr = self.lr
            for stage_epoch in self.stage_list:
                if current_epoch <= stage_epoch:
                    return stage_lr
                else:
                    stage_lr *= self.stage_decay
                pass  # end if
            pass  # end for
            return stage_lr
        elif self.mode == 'linear':
            factor = ratio_epoch
            return self.lr + (self.target_lr - self.lr) * factor
        else:
            raise RuntimeError('Unknown learning rate mode: ' + self.mode)
        pass  # end if


def split_weights(net):
    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)

            if m.bias is not None:
                no_decay.append(m.bias)

        else:
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)

    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def mixup(input, target, alpha=0.2):
    gamma = np.random.beta(alpha, alpha)
    # target is onehot format!
    perm = torch.randperm(input.size(0))
    perm_input = input[perm]
    perm_target = target[perm]
    return input.mul_(gamma).add_(1 - gamma, perm_input), target.mul_(gamma).add_(1 - gamma, perm_target)


def one_hot(y, num_classes, smoothing_eps=None):
    if smoothing_eps is None:
        one_hot_y = F.one_hot(y, num_classes).float()
        return one_hot_y
    else:
        one_hot_y = F.one_hot(y, num_classes).float()
        v1 = 1 - smoothing_eps + smoothing_eps / float(num_classes)
        v0 = smoothing_eps / float(num_classes)
        new_y = one_hot_y * (v1 - v0) + v0
        return new_y


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    cudnn.enabled = True
    seed_torch(args.seed)
    #
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    _IMAGENET_PCA = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }
    lighting_param = 0.1

    class Lighting(object):
        """Lighting noise(AlexNet - style PCA - based noise)"""

        def __init__(self, alphastd, eigval, eigvec):
            self.alphastd = alphastd
            self.eigval = eigval
            self.eigvec = eigvec

        def __call__(self, img):
            if self.alphastd == 0:
                return img

            alpha = img.new().resize_(3).normal_(0, self.alphastd)
            rgb = self.eigvec.type_as(img).clone() \
                .mul(alpha.view(1, 3).expand(3, 3)) \
                .mul(self.eigval.view(1, 3).expand(3, 3)) \
                .sum(1).squeeze()

            return img.add(rgb.view(3, 1, 1).expand_as(img))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_list = [transforms.Resize(256),
                      transforms.RandomResizedCrop(224, interpolation=PIL.Image.BICUBIC),
                      transforms.RandomHorizontalFlip(),
                      # autoaugment.ImageNetPolicy(),
                      transforms.ToTensor(),
                      Lighting(lighting_param, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
                      normalize,
                      hotfix.transforms.RandomErasing()]
    train_transform = transforms.Compose(transform_list)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_data = torchvision.datasets.ImageFolder(root=os.path.join(args.data, 'train'), transform=train_transform)
    valid_data = torchvision.datasets.ImageFolder(root=os.path.join(args.data, 'val'), transform=test_transform)
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=48)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=48)
    num_train_samples = 1281167


    arch = [3, 3, 2, 2, 3, 5, 5, 4, 3, 2, 5, 4, 3, 5, 3, 5, 5, 5, 3, 2, 5]
    model = Network(arch)


    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data, gain=3.26033)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 3.26033 * np.sqrt(2 / (m.weight.shape[0] + m.weight.shape[1])))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    if hasattr(args, 'bn_momentum') and args.bn_momentum is not None:
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.momentum = args.bn_momentum

    if hasattr(args, 'bn_eps') and args.bn_eps is not None:
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eps = args.bn_eps





    input_data = torch.randn(size=(1, 3, 224,224))
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    flops, params = profile(model, inputs=(input_data, ))
    print('FLOPs = ' + str(flops / 1000 ** 2) + 'M')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    if args.parallel:
        model = nn.DataParallel(model.cuda())


    # define loss function (criterion)
    if (hasattr(args, 'label_smoothing') and args.label_smoothing) or (hasattr(args, 'mixup') and args.mixup):
        criterion = cross_entropy
    else:
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

    params = split_weights(model)
    optimizer = torch.optim.SGD(params,
                                args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    if args.load:
        model, optimizer, start_epoch, best_acc_top1 = utils.load_checkpoint(
            model, optimizer,
            '../../experiments/sota/imagenet/eval/EXP-20231025-103641-gz_mobile_net-42-517')
    else:
        best_acc_top1 = 0
        start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        # logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        # model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        logging.info('epoch %d', epoch)
        train_acc, train_obj = train(train_queue, model, criterion, optimizer, epoch, num_train_samples)
        logging.info('train_acc %f', train_acc)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Obj/train', train_obj, epoch)
        # scheduler.step()

        valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc_top1 %f', valid_acc_top1)
        logging.info('valid_acc_top5 %f', valid_acc_top5)
        writer.add_scalar('Acc/valid_top1', valid_acc_top1, epoch)
        writer.add_scalar('Acc/valid_top5', valid_acc_top5, epoch)

        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True

        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc_top1': best_acc_top1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save)


def train(train_queue, model, criterion, optimizer, epoch, num_train_samples=None):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    lr_scheduler = LearningRateScheduler(mode=args.lr_mode,
                                         lr=args.learning_rate,
                                         num_training_instances=num_train_samples,
                                         target_lr=0,
                                         stop_epoch=args.epochs,
                                         warmup_epoch=args.warmup,
                                         stage_list=args.lr_stage_list,
                                         stage_decay=args.lr_stage_decay)
    lr_scheduler.update_lr(batch_size=epoch * num_train_samples)
    optimizer.zero_grad()

    for step, (input, target) in enumerate(train_queue):

        lr_scheduler.update_lr(batch_size=input.shape[0])

        current_lr = lr_scheduler.get_lr()
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        bool_label_smoothing = False
        bool_mixup = False

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            if hasattr(args, 'label_smoothing') and args.label_smoothing:
                bool_label_smoothing = True
            if hasattr(args, 'mixup') and args.mixup:
                bool_mixup = True

            if bool_label_smoothing and not bool_mixup:
                transformed_target = one_hot(target, num_classes=CLASSES, smoothing_eps=0.1)

            if not bool_label_smoothing and bool_mixup:
                transformed_target = one_hot(target, num_classes=CLASSES)
                input, transformed_target = mixup(input, transformed_target)

            if bool_label_smoothing and bool_mixup:
                transformed_target = one_hot(target, num_classes=CLASSES, smoothing_eps=0.1)
                input, transformed_target = mixup(input, transformed_target)

        pass  # end with

        logits = model(input)
        loss = criterion(logits, transformed_target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %f %e %f %f', step, current_lr, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):

            transformed_target = target
            if (hasattr(args, 'label_smoothing') and args.label_smoothing) or (hasattr(args, 'mixup') and args.mixup):
                transformed_target = one_hot(transformed_target, num_classes=CLASSES, smoothing_eps=None)

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            transformed_target = transformed_target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, transformed_target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main()
