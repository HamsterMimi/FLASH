import os

import numpy as np
import torch
import torchvision
from torchvision import transforms

from mobilenet_search_space.retrain_architecture import autoaugment, hotfix
from nasbench201.utils import Lighting, lighting_param, _IMAGENET_PCA


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

if __name__ == '__main__':
    num_train_samples = 1281167
    stop_epoch = 480
    warmup = 5
    lr = 0.1
    target_lr = 0
    mode = 'cosine'
    scheduler = LearningRateScheduler(
        mode,
        lr,
        target_lr=target_lr,
        num_training_instances=num_train_samples,
        stop_epoch=stop_epoch,
        warmup_epoch=5,
        stage_list=None,
        stage_decay=None,
    )
    for epoch in range(480):
        count = 0
        samples = 0
        while True:
            if samples + 256 > num_train_samples:
                up_data = num_train_samples - samples
                scheduler.update_lr(up_data)
                count+=1
                break
            else:
                up_data = 256
                samples += 256
                scheduler.update_lr(up_data)
                curr_lr = scheduler.get_lr()
                count += 1
                if count % 100 == 0:
                    print(epoch, count, curr_lr)

