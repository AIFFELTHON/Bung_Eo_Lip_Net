import torch
import numpy as np


# -- mixup data augmentation # mixup augmentation 계산
# from https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py
def mixup_data(x, y, alpha=1.0, soft_labels = None, use_cuda=False):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''

    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)  # 베타 분포에서 표본 추출
    else:
        lam = 1.

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()  # 주어진 범위 내의 정수를 랜덤하게 생성 # tensor 를 gpu 에 할당
    else:
        index = torch.randperm(batch_size)  # 주어진 범위 내의 정수를 랜덤하게 생성

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# mixup 적용
def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
