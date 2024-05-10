import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
from tqdm import tqdm
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from model_imagenet import GenNetworkImageNet as GenNetwork
from spaces import spaces_dict

parser = argparse.ArgumentParser("training imagenet")
parser.add_argument('--workers', type=int, default=32, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.4, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--pool', type=int, default=10)
parser.add_argument('--arch', type=str, default='PCDARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--warmup', type=int, default=5, help='warmup epochs')
parser.add_argument('--model_name', type=str, default='')
# parser.add_argument('--lr_scheduler', type=str, default='cosine', help='lr scheduler, linear or cosine')
parser.add_argument('--lr_mode', type=str, default='cosine', help='lr scheduler, linear or cosine')
parser.add_argument('--tmp_data_dir', type=str, default='/tmp/cache/', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--measure', type=str, default='meco', choices=['meco', 'zen'])
parser.add_argument('--lr_stage_list', default=None, type=str, help='stage-wise learning epoch list.')
parser.add_argument('--lr_stage_decay', default=None, type=float, help='stage-wise learning epoch list.')
args, unparsed = parser.parse_known_args()

args.save = '../../experiments/sota/imagenet/eval/{}-{}-{}-{}'.format(
    args.save, time.strftime("%Y%m%d-%H%M%S"), args.arch, args.seed)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 1000


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



def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)
    logging.info('PID: ', os.getpid())
    num_gpus = torch.cuda.device_count()
    # genotype = eval("genotypes.%s" % args.arch)
    print('---------Genotype---------')
    # logging.info(genotype)
    print('--------------------------')

    data_dir = args.tmp_data_dir
    traindir = os.path.join(data_dir, 'train')
    validdir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = dset.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    model_name = args.model_name


    if model_name:
        from searched_models import model_dict
        best_model = model_dict[model_name]

    else:
        x = next(iter(train_queue))[0][0].unsqueeze(0)
        s = time.time()
        models_pool = []
        for _ in tqdm(range(args.pool)):
            # model = GenNetwork(arch, args.init_channels, n_classes, args.layers, args.auxiliary, steps=args.steps,
            #                 concat=range(2, 6), data=x,
            #                 primitives=spaces_dict[args.search_space], drop_path_prob=args.drop_path_prob, measure=args.measure)
            model = GenNetwork(args.init_channels, CLASSES, args.layers, args.auxiliary,
                               x=x, primitives=spaces_dict['s5'], drop_path_prob=args.drop_path_prob,
                               measure=args.measure)
            models_pool.append((model, model.get_net_score()))
        e = time.time()
        logging.info('total time: %f', e - s)

        models_id = sorted(range(len(models_pool)), key=lambda k: models_pool[k][1], reverse=True)
        model, idx = models_pool[models_id[0]][0], models_id[0]
        best_model = model.get_arch()
        logging.info(f"model architecture: {best_model}")

    model = GenNetwork(args.init_channels, CLASSES, args.layers, args.auxiliary, x=torch.randn(size=(1, 3, 224, 224)),
                       primitives=spaces_dict['s5'], arch=best_model)

    if num_gpus > 1:
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    logging.info(f'best arch: {best_model}')

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    best_acc_top1 = 0
    best_acc_top5 = 0
    for epoch in range(args.epochs):
        logging.info('Epoch: %d', epoch)
        if num_gpus > 1:
            model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        else:
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        epoch_start = time.time()
        train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer, epoch)
        logging.info('Train_acc: %f', train_acc)

        valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
        logging.info('Valid_acc_top1: %f', valid_acc_top1)
        logging.info('Valid_acc_top5: %f', valid_acc_top5)
        epoch_duration = time.time() - epoch_start
        logging.info('Epoch time: %ds.', epoch_duration)
        is_best = False
        if valid_acc_top5 > best_acc_top5:
            best_acc_top5 = valid_acc_top5
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc_top1': best_acc_top1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save)



def train(train_queue, model, criterion, optimizer, epoch, num_train_samples=1281167):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    batch_time = utils.AvgrageMeter()
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

        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        b_start = time.time()
        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        batch_time.update(time.time() - b_start)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info('TRAIN Step: %03d Lr: %e Objs: %e R1: %f R5: %f Duration: %ds BTime: %.3fs',
                         step, current_lr, objs.avg, top1.avg, top5.avg, duration, batch_time.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits, _ = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info('VALID Step: %03d Objs: %e R1: %f R5: %f Duration: %ds', step, objs.avg, top1.avg, top5.avg,
                         duration)

    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main()