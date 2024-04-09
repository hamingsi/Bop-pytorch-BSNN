import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor): # TBCHW
        y_shape = [x_seq.shape[0], x_seq.shape[1]]  #T*B,C,H,W
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)


class ClassifyLinear(nn.Module):

    def __init__(self, linear, ):
        super(ClassifyLinear, self).__init__()
        self.ops = linear

    def forward(self, x):
        step = x.size(0)
        out = []
        for i in range(step):
            out += [self.ops(x[i])]
        out = torch.stack(out)
        return out

class Layer(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
            nn.BatchNorm2d(out_plane)
        )
        self.act = LIFSpike()

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0, num_bits_u=2):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        # self.num_bits_u = num_bits_u

    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem * self.tau + x[t, ...]
            spike = self.act(mem - self.thresh, self.gama)
            mem = (1 - spike) * mem
            # mem = u_q(mem,self.num_bits_u)
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=0)


def u_q(u, b):
    u = torch.tanh(u)
    alpha = u.data.abs().max()
    u = torch.clamp(u/alpha,min=-1,max=1)
    u = u*(2**(b-1)-1)
    u_hat = (u.round()-u).detach()+u
    return u_hat*alpha/(2**(b-1)-1)

def add_dimention(x, T):
    x.unsqueeze_(0)
    x = x.repeat(T, 1, 1, 1, 1)
    return x


# ----- For ResNet19 code -----


class tdLayer(nn.Module):
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = SeqToANNContainer(layer)
        self.bn = bn

    def forward(self, x):
        x_ = self.layer(x)
        if self.bn is not None:
            x_ = self.bn(x_)
        return x_


class tdBatchNorm(nn.Module):
    def __init__(self, out_panel):
        super(tdBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(out_panel)
        self.seqbn = SeqToANNContainer(self.bn)

    def forward(self, x):
        y = self.seqbn(x)
        return y

"""class myBatchNorm3d(nn.Module):
    def __init__(self, inplanes, step):
        super().__init__()
        self.bn = nn.BatchNorm3d(inplanes)
        self.step = step
    def forward(self, x):
        out = x.permute(1, 2, 0, 3, 4)
        out = self.bn(out)
        out = out.permute(2, 0, 1, 3, 4).contiguous()
        return out"""