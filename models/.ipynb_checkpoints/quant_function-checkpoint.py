import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Function
import torch


# class BinaryQuantize(Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         out = torch.sign(input)
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         # grad_output[input.abs() > 1] = 0
#         return grad_output
class QuantConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.m_t = nn.Parameter(torch.zeros(out_channels,in_channels,kernel_size,kernel_size),requires_grad=False)
        self.register_parameter('m_t', self.m_t)

    def forward(self, input):
        w = self.weight
        # sw = torch.pow(torch.tensor([2]*bw.size(0)).cuda().float(), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(bw.size(0), 1, 1, 1).detach()
        # bw = BinaryQuantize().apply(w)
        # print(w)
        # bw = bw * sw
        output = F.conv2d(input, w, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output

# 8-bit quantization for the first and the last layer
class first_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(first_conv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                         bias)

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q - self.weight).detach() + self.weight
        return F.conv2d(x, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class last_fc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(last_fc, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q - self.weight).detach() + self.weight
        return F.linear(x, weight_q, self.bias)