import os

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.autograd import Variable





# def split_weights(net):
#     """split network weights into to categlories,
#     one are weights in quantconv layer and other layer,
#     others are other learnable paramters(conv bias,
#     bn weights, bn bias, linear bias)

#     Args:
#         net: network architecture

#     Returns:
#         a dictionary of params splite into to categlories
#     """

#     decay = []
#     no_decay = []

#     for m in net.modules():
#         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#             decay.append(m.weight)
#             #print("hello")
#             #print(m)

#             if m.bias is not None:
#                 no_decay.append(m.bias)

#         else:
#             if hasattr(m, 'weight'):
#                 no_decay.append(m.weight)
#             if hasattr(m, 'bias'):
#                 no_decay.append(m.bias)
#             # if hasattr(m, 'alpha'):
#                 # quant alpha
#                 # no_decay.append(m.alpha)

#             #print("buhello")
#             #print(m)

#     assert len(list(net.parameters())) == len(decay) + len(no_decay)

#     return [dict(params=decay), dict(params=no_decay, weight_decay=0)]