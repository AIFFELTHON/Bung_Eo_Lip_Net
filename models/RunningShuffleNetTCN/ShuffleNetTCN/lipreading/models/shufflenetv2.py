import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math

import pdb  # 파이썬 디버거


# Conv2D (3,3) + BatchNorm2D + ReLU
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


# Conv2D (1,1) + BatchNorm2D + ReLU
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


# reshape -> flatten
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()  # data 정보

    channels_per_group = num_channels // groups  # 그룹당 채널 계산
    
    # reshape
    x = x.view(batchsize, groups, # reshape 적용된 모양의 tensor 반환 # 원본 data 공유
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()  # transpose(): 2개의 차원 맞교환 # contiguous(): 원본과 다른 새로운 주소로 할당

    # flatten => [batchsize, height * width]
    x = x.view(batchsize, -1, height, width)  # reshape 적용된 모양의 tensor 반환 # 원본 data 공유

    return x
    

# Inverted Residual - 관련 모델: MobileNetV2
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride

        # stride 가 [1,2] 인지 확인, 아니면 AssertionError 메시지를 띄움
        assert stride in [1, 2]  # 원하는 조건의 변수값을 보증하기 위해 사용

        oup_inc = oup//2
        
        if self.benchmodel == 1:
            #assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),  # Conv2D (1,1)
                nn.BatchNorm2d(oup_inc),  # BatchNorm2D
                nn.ReLU(inplace=True),  # ReLU
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),  # Conv2D (3,3)
                nn.BatchNorm2d(oup_inc),  # BatchNorm2D
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),  # Conv2D (1,1)
                nn.BatchNorm2d(oup_inc),  # BatchNorm2D
                nn.ReLU(inplace=True),  # ReLU
            )                
        else:                  
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),  # Conv2D (3,3)
                nn.BatchNorm2d(inp),  # BatchNorm2D
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),  # Conv2D (1,1)
                nn.BatchNorm2d(oup_inc),  # BatchNorm2D
                nn.ReLU(inplace=True),  # ReLU
            )        
    
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),  # Conv2D (1,1)
                nn.BatchNorm2d(oup_inc),  # BatchNorm2D
                nn.ReLU(inplace=True),  # ReLU
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),  # Conv2D (3,3)
                nn.BatchNorm2d(oup_inc),  # BatchNorm2D
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),  # Conv2D (1,1)
                nn.BatchNorm2d(oup_inc),  # BatchNorm2D
                nn.ReLU(inplace=True),  # ReLU
            )
          
    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)  # Tensor list를 한번에 tensor로 만들기

    # 모델이 학습데이터를 입력받아서 forward propagation 진행
    def forward(self, x):
        if 1==self.benchmodel:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2==self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)  # reshape -> flatten


# 셔플넷 V2
class ShuffleNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=2.):
        super(ShuffleNetV2, self).__init__()
        
        # 인풋사이즈 % 32 == 0 인지 확인, 아니면 AssertionError 메시지를 띄움
        assert input_size % 32 == 0, "Input size needs to be divisible by 32"  # 원하는 조건의 변수값을 보증하기 위해 사용
        
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24,  48,  96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise ValueError(  # 에러 발생시키기
                """Width multiplier should be in [0.5, 1.0, 1.5, 2.0]. Current value: {}""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2)  # Conv2D (3,3) + BatchNorm2D + ReLU
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # MaxPool2D
        
        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]
            for i in range(numrepeat):
                if i == 0:
                #inp, oup, stride, benchmodel):
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel
                
                
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building last several layers
        self.conv_last  = conv_1x1_bn(input_channel, self.stage_out_channels[-1])  # Conv2D (1,1) + BatchNorm2D + ReLU
        self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size/32)))  # AvgPool2D              
        
        # building classifier # 선형 회귀 모델
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))

    # 모델이 학습데이터를 입력받아서 forward propagation 진행
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        x = x.view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x
