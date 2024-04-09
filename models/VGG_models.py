import random
from models.layers import *


from models.quant_function import *
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from models.layers import *

class VGG11_BSNN(nn.Module):

    def __init__(self, num_classes=10,step=2,num_b=2,input_c=3):
        super(VGG11_BSNN, self).__init__()
        self.T = step
        self.num_b = num_b
        self.conv0 = first_conv(input_c, 64, kernel_size=3, padding=1, bias=False)
        self.bn0 = tdBatchNorm(64)
        self.conv0_s = tdLayer(self.conv0,self.bn0)
        self.lif0 = LIFSpike(num_bits_u=self.num_b)

        self.conv1 = QuantConv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = tdBatchNorm(128)
        # self.nonlinear = nn.ReLU(inplace=True)
        self.conv1_s = tdLayer(self.conv1, self.bn1)
        self.lif1 = LIFSpike(num_bits_u=self.num_b)

        self.pooling = SeqToANNContainer(nn.MaxPool2d(kernel_size=2,stride=2)) 
        
        self.conv2 = QuantConv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = tdBatchNorm(256)
        self.conv2_s = tdLayer(self.conv2, self.bn2)
        self.lif2 = LIFSpike(num_bits_u=self.num_b)

        self.conv3 = QuantConv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = tdBatchNorm(256)
        self.conv3_s = tdLayer(self.conv3, self.bn3)
        self.lif3 = LIFSpike(num_bits_u=self.num_b)

        self.conv4 = QuantConv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = tdBatchNorm(512)
        self.conv4_s = tdLayer(self.conv4, self.bn4)
        self.lif4 = LIFSpike(num_bits_u=self.num_b)

        self.conv5 = QuantConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = tdBatchNorm(512)
        self.conv5_s = tdLayer(self.conv5, self.bn5)
        self.lif5 = LIFSpike(num_bits_u=self.num_b)

        self.conv6 = QuantConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn6 = tdBatchNorm(512)
        self.conv6_s = tdLayer(self.conv6, self.bn6)
        self.lif6 = LIFSpike(num_bits_u=self.num_b)

        self.conv7 = QuantConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn7 = tdBatchNorm(512)
        self.conv7_s = tdLayer(self.conv7, self.bn7)
        self.lif7 = LIFSpike(num_bits_u=self.num_b)

        # W = int(48/2/2/2/2)
        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))

        self.fc = ClassifyLinear(last_fc(512, num_classes))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, QuantConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def forward(self, x):
        fr = []
        x = add_dimention(x, self.T)
        x = self.conv0_s(x)
        x = self.lif0(x)
        fr.append(x.mean())
        
        x = self.conv1_s(x)
        x = self.lif1(x)
        fr.append(x.mean())

        x = self.pooling(x)

        x = self.conv2_s(x)
        x = self.lif2(x)
        fr.append(x.mean())

        x = self.conv3_s(x)
        x = self.lif3(x)
        fr.append(x.mean())

        x = self.pooling(x)

        x = self.conv4_s(x)
        x = self.lif4(x)
        fr.append(x.mean())

        x = self.conv5_s(x)
        x = self.lif5(x)
        fr.append(x.mean())

        x = self.pooling(x)

        x = self.conv6_s(x)
        x = self.lif6(x)
        fr.append(x.mean())

        x = self.conv7_s(x)
        x = self.lif7(x)
        fr.append(x.mean())

        x = self.avgpool(x)
        
        x = torch.flatten(x, 2)
        x = self.fc(x)
        return x, fr
    

class VGG11_BSNN(nn.Module):

    def __init__(self, num_classes=10,step=2,num_b=2,input_c=3):
        super(VGG11_BSNN, self).__init__()
        self.T = step
        self.num_b = num_b
        self.conv0 = first_conv(input_c, 64, kernel_size=3, padding=1, bias=False)
        self.bn0 = tdBatchNorm(64)
        self.conv0_s = tdLayer(self.conv0,self.bn0)
        self.lif0 = LIFSpike(num_bits_u=self.num_b)

        self.conv1 = QuantConv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = tdBatchNorm(128)
        # self.nonlinear = nn.ReLU(inplace=True)
        self.conv1_s = tdLayer(self.conv1, self.bn1)
        self.lif1 = LIFSpike(num_bits_u=self.num_b)

        self.pooling = SeqToANNContainer(nn.MaxPool2d(kernel_size=2,stride=2)) 
        
        self.conv2 = QuantConv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = tdBatchNorm(256)
        self.conv2_s = tdLayer(self.conv2, self.bn2)
        self.lif2 = LIFSpike(num_bits_u=self.num_b)

        self.conv3 = QuantConv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = tdBatchNorm(256)
        self.conv3_s = tdLayer(self.conv3, self.bn3)
        self.lif3 = LIFSpike(num_bits_u=self.num_b)

        self.conv4 = QuantConv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = tdBatchNorm(512)
        self.conv4_s = tdLayer(self.conv4, self.bn4)
        self.lif4 = LIFSpike(num_bits_u=self.num_b)

        self.conv5 = QuantConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = tdBatchNorm(512)
        self.conv5_s = tdLayer(self.conv5, self.bn5)
        self.lif5 = LIFSpike(num_bits_u=self.num_b)

        self.conv6 = QuantConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn6 = tdBatchNorm(512)
        self.conv6_s = tdLayer(self.conv6, self.bn6)
        self.lif6 = LIFSpike(num_bits_u=self.num_b)

        self.conv7 = QuantConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn7 = tdBatchNorm(512)
        self.conv7_s = tdLayer(self.conv7, self.bn7)
        self.lif7 = LIFSpike(num_bits_u=self.num_b)

        # W = int(48/2/2/2/2)
        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))

        self.fc = ClassifyLinear(last_fc(512, num_classes))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, QuantConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def forward(self, x):
        fr = []
        x = add_dimention(x, self.T)
        x = self.conv0_s(x)
        x = self.lif0(x)
        fr.append(x.mean())
        
        x = self.conv1_s(x)
        x = self.lif1(x)
        fr.append(x.mean())

        x = self.pooling(x)

        x = self.conv2_s(x)
        x = self.lif2(x)
        fr.append(x.mean())

        x = self.conv3_s(x)
        x = self.lif3(x)
        fr.append(x.mean())

        x = self.pooling(x)

        x = self.conv4_s(x)
        x = self.lif4(x)
        fr.append(x.mean())

        x = self.conv5_s(x)
        x = self.lif5(x)
        fr.append(x.mean())

        x = self.pooling(x)

        x = self.conv6_s(x)
        x = self.lif6(x)
        fr.append(x.mean())

        x = self.conv7_s(x)
        x = self.lif7(x)
        fr.append(x.mean())

        x = self.avgpool(x)
        
        x = torch.flatten(x, 2)
        x = self.fc(x)
        return x, fr