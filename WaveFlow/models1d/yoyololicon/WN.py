import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, set_grad_enabled, grad, gradcheck
from efficient_util import add_weight_norms
import numpy as np

from functools import reduce
from operator import mul


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(zw, zf):
    t_act = torch.tanh(zw)
    s_act = torch.sigmoid(zf)
    acts = t_act * s_act
    return acts


class _NonCausalLayer(nn.Module):
    def __init__(self,
                 dilation,
                 dilation_channels,
                 residual_channels,
                 skip_channels,
                 aux_channels,
                 radix,
                 bias,
                 last_layer=False):
        super().__init__()
        pad_size = dilation * (radix - 1) // 2
        self.WV = nn.Conv1d(residual_channels + aux_channels, dilation_channels * 2, kernel_size=radix,
                            padding=pad_size, dilation=dilation, bias=bias)
        
        self.chs_split = [skip_channels]
        if last_layer:
            self.W_o = nn.Conv1d(dilation_channels, skip_channels, 1, bias=bias)
        else:
            self.W_o = nn.Conv1d(dilation_channels, residual_channels + skip_channels, 1, bias=bias)
            self.chs_split.insert(0, residual_channels)
    
    def forward(self, audio, spect):
        xy = torch.cat((audio, spect), 1) # add along channel dim
                              #[B, n_group//2, T//n_group] + [B, mels, T//n_group] -> [B, n_group//2+mels, T//n_group]
        zw, zf = self.WV(xy).chunk(2, 1) # split along channel dim -> 
        
        z = fused_add_tanh_sigmoid_multiply(zw, zf)
        
        *z, skip = self.W_o(z).split(self.chs_split, 1)
        return z[0] + audio if len(z) else None, skip
    
    
class WN(nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary difference
    from WaveNet is the convolutions need not be causal.  There is also no dilation
    size reset.  The dilation only doubles on each layer
    """
    def __init__(self, in_channels, aux_channels, dilation_channels=256,
                 residual_channels=256, skip_channels=256, n_layers=8,
                 radix=3, bias=False, zero_init=True):
        super().__init__()
        dilations = 2 ** torch.arange(n_layers)
        self.dilations = dilations.tolist()
        self.in_chs = in_channels
        self.res_chs = residual_channels
        self.dil_chs = dilation_channels
        self.skp_chs = skip_channels
        self.aux_chs = aux_channels
        self.rdx = radix
        self.r_field = sum(self.dilations) + 1
        
        self.start = nn.Conv1d(in_channels, residual_channels, 1, bias=bias)
        self.start.apply(add_weight_norms)
        
        
        self.layers = nn.ModuleList(_NonCausalLayer(d,
                                                    dilation_channels,
                                                    residual_channels,
                                                    skip_channels,
                                                    aux_channels,
                                                    radix,
                                                    bias) for d in self.dilations[:-1])
        self.layers.append(_NonCausalLayer(self.dilations[-1],
                                           dilation_channels,
                                           residual_channels,
                                           skip_channels,
                                           aux_channels,
                                           radix,
                                           bias,
                                           last_layer=True))
        self.layers.apply(add_weight_norms)
        
        self.end = nn.Conv1d(skip_channels, in_channels * 2, 1, bias=bias)
        if zero_init:
            self.end.weight.data.zero_()
            if bias:
                self.end.bias.data.zero_()
    
    def forward(self, audio, spect):
        audio = self.start(audio)
        
        cum_skip = None
        for layer in self.layers:
            audio, skip = layer(audio, spect)
            if cum_skip is None:
                cum_skip = skip
            else:
                cum_skip += skip
        return self.end(cum_skip).chunk(2, 1)