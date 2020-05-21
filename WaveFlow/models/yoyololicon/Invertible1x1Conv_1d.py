import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, set_grad_enabled, grad, gradcheck
from efficient_util import add_weight_norms
import numpy as np

from functools import reduce
from operator import mul


class Invertible1x1Conv(nn.Conv1d):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """
    def __init__(self, c, hparams):
        super().__init__(c, c, 1, bias=False) # init as nn.Conv1d(c, c, kernel_size=1, stride=1) 
        memory_efficient = hparams.InvConv_memory_efficient
        
        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]
        
        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:,0] = -1*W[:,0]
        W = W.view(c, c, 1)
        self.weight.data = W
    
        if memory_efficient:
            self.efficient_forward = Conv1x1Func.apply
            self.efficient_inverse = InvConv1x1Func.apply

    def forward(self, z):
        if hasattr(self, 'efficient_forward'):
            audio_out, log_det_W = self.efficient_forward(z, self.weight)
            z.storage().resize_(0)
            return audio_out, log_det_W
        else:
            *_, n_of_groups = z.shape
            log_det_W = n_of_groups * self.weight.squeeze().slogdet()[1]  # should fix nan logdet
            audio_out = super().forward(z)
            return audio_out, log_det_W
    
    def inverse(self, audio_out):
        if hasattr(self, 'efficient_inverse'):
            z, log_det_W = self.efficient_inverse(audio_out, self.weight)
            audio_out.storage().resize_(0)
            return z, log_det_W
        else:
            weight = self.weight.squeeze()
            *_, n_of_groups = audio_out.shape
            log_det_W = -n_of_groups * weight.slogdet()[1]  # should fix nan logdet
            z = F.conv1d(audio_out, weight.inverse().unsqueeze(-1))
            return z, log_det_W


class Conv1x1Func(Function):
    @staticmethod
    def forward(ctx, z, weight):
        with torch.no_grad():
            *_, n_of_groups = z.shape
            log_det_W = n_of_groups * weight.squeeze().slogdet()[1]
            #log_det_W = n_of_groups * weight.squeeze().float().slogdet()[1].half()
            audio_out = F.conv1d(z, weight)
        
        ctx.save_for_backward(z.data, weight, audio_out)
        return audio_out, log_det_W
    
    @staticmethod
    def backward(ctx, z_grad, log_det_W_grad):
        z, weight, audio_out = ctx.saved_tensors
        *_, n_of_groups = audio_out.shape
        
        with torch.no_grad():
            inv_weight = weight.squeeze().inverse()
            #inv_weight = weight.squeeze().float().inverse().half()
            z.storage().resize_(reduce(mul, audio_out.shape))
            z[:] = F.conv1d(audio_out, inv_weight.unsqueeze(-1))
            
            dx = F.conv1d(z_grad, weight[..., 0].t().unsqueeze(-1))
            dw = z_grad.transpose(0, 1).contiguous().view(weight.shape[0], -1) @ z.transpose(1, 2).contiguous().view(
                -1, weight.shape[1])
            dw += inv_weight.t() * log_det_W_grad * n_of_groups
        
        return dx, dw.unsqueeze(-1)


class InvConv1x1Func(Function):
    @staticmethod
    def forward(ctx, z, inv_weight):
        with torch.no_grad():
            sqr_inv_weight = inv_weight.squeeze()
            *_, n_of_groups = z.shape
            log_det_W = -sqr_inv_weight.slogdet()[1]
            log_det_W *= n_of_groups
            audio_out = F.conv1d(z, sqr_inv_weight.inverse().unsqueeze(-1))
        
        ctx.save_for_backward(z.data, inv_weight, audio_out)
        return audio_out, log_det_W
    
    @staticmethod
    def backward(ctx, z_grad, log_det_W_grad):
        z, inv_weight, audio_out = ctx.saved_tensors
        *_, n_of_groups = audio_out.shape
        
        with torch.no_grad():
            z.storage().resize_(reduce(mul, audio_out.shape))
            z[:] = F.conv1d(audio_out, inv_weight)
            
            inv_weight = inv_weight.squeeze()
            weight_T = inv_weight.inverse().t()
            dx = F.conv1d(z_grad, weight_T.unsqueeze(-1))
            dw = z_grad.transpose(0, 1).contiguous().view(weight_T.shape[0], -1) @ \
                 z.transpose(1, 2).contiguous().view(-1, weight_T.shape[1])
            dinvw = - weight_T @ dw @ weight_T
            dinvw -= weight_T * log_det_W_grad * n_of_groups
        
        return dx, dinvw.unsqueeze(-1)