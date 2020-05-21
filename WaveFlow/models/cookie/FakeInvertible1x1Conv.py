import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, set_grad_enabled, grad, gradcheck
from efficient_util import add_weight_norms
import numpy as np

from functools import reduce
from operator import mul


class Invertible1x1Conv(nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """
    def __init__(self, hparams):
        super(Invertible1x1Conv).__init__()
    
    def forward(self, z):
        log_det_W = None
        return z, log_det_W
    
    def inverse(self, audio_out):
        log_det_W = None
        return z, log_det_W