import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Function
import torch
from models.quant_function import *
from models.layers import *

def l_prod(in_list):
    res = 1
    for _ in in_list:
        res *= _
    return res

def calculate_conv2d_flops(input_size: list, output_size: list, kernel_size: list, groups: int, bias: bool = False):
    n, out_c, oh, ow = output_size
    n, in_c, ih, iw = input_size
    out_c, in_c, kh, kw = kernel_size
    in_c = input_size[1]
    g = groups
    return l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])

def calculate_lif_flops(x, y):
    input_size = list(x.shape)
    output_size = list(y.shape)
    return l_prod(output_size)