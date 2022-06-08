import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import pdb


"""Implements Temporal Convolutional Network (TCN)

__https://arxiv.org/pdf/1803.01271.pdf
"""

# Casual Conv1D
class Chomp1d(nn.Module):
    def __init__(self, chomp_size, symm_chomp):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symm_chomp = symm_chomp
        if self.symm_chomp:
            assert self.chomp_size % 2 == 0, "If symmetric chomp, chomp size needs to be even"
      
    # 모델이 학습데이터를 입력받아서 forward propagation 진행
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        if self.symm_chomp:
            return x[:, :, self.chomp_size//2:-self.chomp_size//2].contiguous()
        else:
            return x[:, :, :-self.chomp_size].contiguous()
        

# Conv1D + BatchNorm1D + Casual Conv1D + ReLU
class ConvBatchChompRelu(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, relu_type, dwpw=False):
        super(ConvBatchChompRelu, self).__init__()
        self.dwpw = dwpw
        if dwpw:
            self.conv = nn.Sequential(
                # -- dw
                nn.Conv1d( n_inputs, n_inputs, kernel_size, stride=stride,  # Conv1D
                           padding=padding, dilation=dilation, groups=n_inputs, bias=False),
                nn.BatchNorm1d(n_inputs),  # BatchNorm1D
                Chomp1d(padding, True),  # Casual Conv1D
                nn.PReLU(num_parameters=n_inputs) if relu_type == 'prelu' else nn.ReLU(inplace=True),  # PReLU or ReLU
                # -- pw
                nn.Conv1d( n_inputs, n_outputs, 1, 1, 0, bias=False),  # Conv1D
                nn.BatchNorm1d(n_outputs),  # BatchNorm1D
                nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU(inplace=True)  # PReLU or ReLU
            )
        else:
            self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size,  # Conv1D
                                               stride=stride, padding=padding, dilation=dilation)
            self.batchnorm = nn.BatchNorm1d(n_outputs)  # BatchNorm1D
            self.chomp = Chomp1d(padding,True)  # Casual Conv1D
            self.non_lin = nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU()  # PReLU or ReLU

    # 모델이 학습데이터를 입력받아서 forward propagation 진행
    def forward(self, x):
        if self.dwpw:
            return self.conv(x)
        else:
            out = self.conv( x )
            out = self.batchnorm( out )
            out = self.chomp( out )
            return self.non_lin( out )


        
# --------- MULTI-BRANCH VERSION ---------------
class MultibranchTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_sizes, stride, dilation, padding, dropout=0.2, 
                 relu_type = 'relu', dwpw=False):
        super(MultibranchTemporalBlock, self).__init__()
        
        self.kernel_sizes = kernel_sizes
        self.num_kernels = len( kernel_sizes )
        self.n_outputs_branch = n_outputs // self.num_kernels
        assert n_outputs % self.num_kernels == 0, "Number of output channels needs to be divisible by number of kernels"



        for k_idx,k in enumerate( self.kernel_sizes ):
            cbcr = ConvBatchChompRelu( n_inputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx], relu_type, dwpw=dwpw)  # Conv1D + BatchNorm1D + Casual Conv1D + ReLU
            setattr( self,'cbcr0_{}'.format(k_idx), cbcr )  # object 에 존재하는 속성의 값을 바꾸거나 새로운 속성을 생성하여 값을 부여함
        self.dropout0 = nn.Dropout(dropout)  # Dropout
        
        for k_idx,k in enumerate( self.kernel_sizes ):
            cbcr = ConvBatchChompRelu( n_outputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx], relu_type, dwpw=dwpw)  # Conv1D + BatchNorm1D + Casual Conv1D + ReLU
            setattr( self,'cbcr1_{}'.format(k_idx), cbcr )  # object 에 존재하는 속성의 값을 바꾸거나 새로운 속성을 생성하여 값을 부여함
        self.dropout1 = nn.Dropout(dropout)  # Dropout

        # downsample?
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if (n_inputs//self.num_kernels) != n_outputs else None  # Conv1D or None
        
        # final relu
        if relu_type == 'relu':
            self.relu_final = nn.ReLU()  # ReLU
        elif relu_type == 'prelu':
            self.relu_final = nn.PReLU(num_parameters=n_outputs)  # PReLU

    # 모델이 학습데이터를 입력받아서 forward propagation 진행
    def forward(self, x):

        # first multi-branch set of convolutions
        outputs = []
        for k_idx in range( self.num_kernels ):
            branch_convs = getattr(self,'cbcr0_{}'.format(k_idx))
            outputs.append( branch_convs(x) )
        out0 = torch.cat(outputs, 1)
        out0 = self.dropout0( out0 )

        # second multi-branch set of convolutions
        outputs = []
        for k_idx in range( self.num_kernels ):
            branch_convs = getattr(self,'cbcr1_{}'.format(k_idx))
            outputs.append( branch_convs(out0) )
        out1 = torch.cat(outputs, 1)
        out1 = self.dropout1( out1 )
                
        # downsample?
        res = x if self.downsample is None else self.downsample(x)

        return self.relu_final(out1 + res)

class MultibranchTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, tcn_options, dropout=0.2, relu_type='relu', dwpw=False):
        super(MultibranchTemporalConvNet, self).__init__()

        self.ksizes = tcn_options['kernel_size']

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]


            padding = [ (s-1)*dilation_size for s in self.ksizes]            
            layers.append( MultibranchTemporalBlock( in_channels, out_channels, self.ksizes, 
                stride=1, dilation=dilation_size, padding = padding, dropout=dropout, relu_type = relu_type,
                dwpw=dwpw) )

        self.network = nn.Sequential(*layers)  # 설정한 레이어 반환

    # 모델이 학습데이터를 입력받아서 forward propagation 진행
    def forward(self, x):
        return self.network(x)        
# --------------------------------


# --------------- STANDARD VERSION (SINGLE BRANCH) ------------------------
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, 
                 symm_chomp = False, no_padding = False, relu_type = 'relu', dwpw=False):
        super(TemporalBlock, self).__init__()
        
        self.no_padding = no_padding
        if self.no_padding:
            downsample_chomp_size = 2*padding-4
            padding = 1 # hack-ish thing so that we can use 3 layers

        if dwpw:
            self.net = nn.Sequential(
                # -- first conv set within block
                # -- dw
                nn.Conv1d( n_inputs, n_inputs, kernel_size, stride=stride,  # Conv1D
                           padding=padding, dilation=dilation, groups=n_inputs, bias=False),
                nn.BatchNorm1d(n_inputs),  # BatchNorm1D
                Chomp1d(padding, True),  # Casual Conv1D
                nn.PReLU(num_parameters=n_inputs) if relu_type == 'prelu' else nn.ReLU(inplace=True),  # PReLU or ReLU
                # -- pw
                nn.Conv1d( n_inputs, n_outputs, 1, 1, 0, bias=False),  # Conv1D (1,1)
                nn.BatchNorm1d(n_outputs),  # BatchNorm1D
                nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU(inplace=True),  # PReLU or ReLU
                nn.Dropout(dropout),  # Dropout
                # -- second conv set within block
                # -- dw
                nn.Conv1d( n_outputs, n_outputs, kernel_size, stride=stride,  # Conv1D
                           padding=padding, dilation=dilation, groups=n_outputs, bias=False),
                nn.BatchNorm1d(n_outputs),  # BatchNorm1D
                Chomp1d(padding, True),  # Casual Conv1D
                nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU(inplace=True),  # PReLU or ReLU
                # -- pw
                nn.Conv1d( n_outputs, n_outputs, 1, 1, 0, bias=False),  # Conv1D
                nn.BatchNorm1d(n_outputs),  # BatchNorm1D
                nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU(inplace=True),  # PReLU or ReLU
                nn.Dropout(dropout),  # Dropout
            )
        else:
            self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,  # Conv1D
                                   stride=stride, padding=padding, dilation=dilation)
            self.batchnorm1 = nn.BatchNorm1d(n_outputs)  # BatchNorm1D
            self.chomp1 = Chomp1d(padding,symm_chomp)  if not self.no_padding else None  # Casual Conv1D or None
            if relu_type == 'relu':
                self.relu1 = nn.ReLU()  # ReLU
            elif relu_type == 'prelu':
                self.relu1 = nn.PReLU(num_parameters=n_outputs)  # PReLU
            self.dropout1 = nn.Dropout(dropout)  # Dropout
            
            self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,  # Conv1D
                                               stride=stride, padding=padding, dilation=dilation)
            self.batchnorm2 = nn.BatchNorm1d(n_outputs)  # BatchNorm1D
            self.chomp2 = Chomp1d(padding,symm_chomp) if not self.no_padding else None  # Casual Conv1D or None
            if relu_type == 'relu':
                self.relu2 = nn.ReLU()  # ReLU
            elif relu_type == 'prelu':
                self.relu2 = nn.PReLU(num_parameters=n_outputs)  # PReLU
            self.dropout2 = nn.Dropout(dropout)  # Dropout
            
      
            if self.no_padding:
                self.net = nn.Sequential(self.conv1, self.batchnorm1, self.relu1, self.dropout1,
                                         self.conv2, self.batchnorm2, self.relu2, self.dropout2)
            else:
                self.net = nn.Sequential(self.conv1, self.batchnorm1, self.chomp1, self.relu1, self.dropout1,
                                         self.conv2, self.batchnorm2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None  # Conv1D or None
        if self.no_padding:
            self.downsample_chomp = Chomp1d(downsample_chomp_size,True)  # Casual Conv1D
        if relu_type == 'relu':
            self.relu = nn.ReLU()  # ReLU
        elif relu_type == 'prelu':
            self.relu = nn.PReLU(num_parameters=n_outputs)  # PReLU

    # 모델이 학습데이터를 입력받아서 forward propagation 진행
    def forward(self, x):
        out = self.net(x)
        if self.no_padding:
            x = self.downsample_chomp(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# TCN 모델
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, tcn_options, dropout=0.2, relu_type='relu', dwpw=False):
        super(TemporalConvNet, self).__init__()
        self.ksize = tcn_options['kernel_size'][0] if isinstance(tcn_options['kernel_size'], list) else tcn_options['kernel_size']
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append( TemporalBlock(in_channels, out_channels, self.ksize, stride=1, dilation=dilation_size,
                                     padding=(self.ksize-1) * dilation_size, dropout=dropout, symm_chomp = True,
                                     no_padding = False, relu_type=relu_type, dwpw=dwpw) )
            
        self.network = nn.Sequential(*layers)  # 설정한 레이어 반환

    # 모델이 학습데이터를 입력받아서 forward propagation 진행
    def forward(self, x):
        return self.network(x)
# --------------------------------
