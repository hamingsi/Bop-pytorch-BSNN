import torch
from torch.optim import Optimizer
from models.quant_function import QuantConv2d
import torch.nn as nn


def split_weights(model):
    """split network weights into to categlories,
    one are weights in quantconv layer and other layer,
    others are other learnable paramters(conv bias,
    bn weights, bn bias, linear bias)

    Args:
        net: network architecture

    Returns:
        a dictionary of params splite into to categlories
    """
    quant_conv = []
    qconv_mt = []
    decay = []
    no_decay = []

    for m in model.modules():
        if isinstance(m, QuantConv2d):
            # if m.weight == 0:
            #     print(m.weight)
            quant_conv.append(m.weight)
            qconv_mt.append(m.m_t)
        elif isinstance(m, nn.Conv2d)or isinstance(m, nn.Linear):
            decay.append(m.weight)
            if m.bias is not None:
                no_decay.append(m.bias)
        else:
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)
    assert len(list(model.parameters())) == len(decay) + len(no_decay) + len(quant_conv) + len(qconv_mt)

    return [dict(params=quant_conv,m_t=qconv_mt,quant=True),dict(params=decay,quant=False), dict(params=no_decay, weight_decay=0,quant=False)]

class Bop(Optimizer): 
    """self design Bop from Rethinking Binarized Neural Network Optimization arXiv:1906.02107,
    Need to implement method to flip weight to avoid updating latent weight.
    Args:
        params(iterable): iterable of parameters to optimize or dicts defining parameter group
        lr(float): learning rate for non-quantcConv
        threshold: determines to whether to flip each weight.
        gamma: the adaptivity rate.
    Example:
        >>> from Bop import *
        >>> optimizer = Bop(model.parameters(), lr=0.1,  threshold=1e-7, gamma=1e-2)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
    def __init__(self, params, lr=0.1, threshold=1e-6, gamma=1e-3,weight_decay=0):
        defaults = dict(lr=lr, threshold=threshold,gamma=gamma,weight_decay=weight_decay)
        super(Bop,self).__init__(params,defaults=defaults)
    
    def __setstate__(self, state):
        super(Bop, self).__setstate__(state)

    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            if group["quant"]:
                for p,p_mt in zip(group['params'], group["m_t"]):
                    d_p = p.grad.data
                    if p.grad is None:
                        continue
                    if group['quant']:
                        # 对 quant2d 权重进行特别处理
                        p_mt.copy_((1 - group['gamma']) * p_mt + group['gamma'] * d_p)
                        p.data.copy_(torch.sign(-torch.sign(p.data.clone() * p_mt - group['threshold']) * p.data.clone())) 
            else:
                for p in group['params']:
                    d_p = p.grad.data
                    if p.grad is None:
                        continue
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    p.data.add_(-group['lr'], d_p)
        return loss
