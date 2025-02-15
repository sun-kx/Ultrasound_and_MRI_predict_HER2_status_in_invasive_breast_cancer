'''
Author: Bercy
Date: 2022-10-17 16:18:26
LastEditors: Bercy
LastEditTime: 2022-10-18 16:12:14
'''
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/10/17 16:50:41
@Author  :   Bercy 
@Version :   1.0
@Contact :   hebingxi0616@163.com
'''

import os
import numpy as np
import torch
from sklearn import metrics 
from sklearn import metrics

def get_result_binary(output_all, output_pred_all, label_all):
    acc = metrics.accuracy_score(label_all.cpu(), output_pred_all.cpu())
    auc = metrics.roc_auc_score(label_all.cpu(), output_all.cpu(), multi_class='ovr')
    precision = metrics.precision_score(label_all.cpu(), output_pred_all.cpu(), average='binary')
    recall = metrics.recall_score(label_all.cpu(), output_pred_all.cpu(), average='binary')
    f1 = metrics.f1_score(label_all.cpu(), output_pred_all.cpu(), average='binary')

    return [acc, auc, precision, recall, f1]

def adjust_learning_rate(optimizer, epoch, lr, lr_decay_rate, warm_epoch=10, warmup=False):
    ''' Adjusts learning rate according to (epoch, lr and lr_decay_rate)

    :param optimizer: (torch.optim object)
    :param epoch: (int)
    :param lr: (float) the initial learning rate
    :param lr_decay_rate: (float) learning rate decay rate
    :return lr_: (float) updated learning rate
    '''
    if warmup:
        if epoch < warm_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr / (warm_epoch-epoch)
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr / (1+(epoch-warm_epoch)*lr_decay_rate)
        #print(lr / (warm_epoch-epoch))
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr / (1+epoch*lr_decay_rate)
    return optimizer.param_groups[0]['lr']