import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, set_grad_enabled, grad, gradcheck
from efficient_util import add_weight_norms
import numpy as np

from functools import reduce
from operator import mul


class AffineCouplingBlock(nn.Module):
    def __init__(self, transform_type, hparams):
        super().__init__()
        
        self.WN = transform_type(hparams)
        if hparams.ACB_memory_efficient:
            self.efficient_forward = AffineCouplingFunc.apply
            self.efficient_inverse = InvAffineCouplingFunc.apply
            self.param_list = list(self.WN.parameters())
    
    def forward(self, z, spect, speaker_ids):
        if hasattr(self, 'efficient_forward'):
            audio_out, log_s = self.efficient_forward(z, spect, speaker_ids, self.WN, *self.param_list)
            z.storage().resize_(0)
            return audio_out, log_s
        else:
            audio_0, audio_1 = z.chunk(2, 1)
            audio_0_out = audio_0
            log_s, t = self.WN(audio_0, spect, speaker_ids)
            audio_1_out = audio_1 * log_s.exp() + t
            audio_out = torch.cat((audio_0_out, audio_1_out), 1)
            return audio_out, log_s
    
    def inverse(self, audio_out, spect, speaker_ids):
        if hasattr(self, 'efficient_inverse'):
            z, log_s = self.efficient_inverse(audio_out, spect, speaker_ids, self.WN, *self.param_list)
            audio_out.storage().resize_(0)
            return z, log_s
        else:
            audio_0_out, audio_1_out = audio_out.chunk(2, 1)
            audio_0 = audio_0_out
            log_s, t = self.WN(audio_0_out, spect, speaker_ids)
            audio_1 = (audio_1_out - t) / log_s.exp()
            z = torch.cat((audio_0, audio_1), 1)
            return z, -log_s


class AffineCouplingFunc(Function):
    @staticmethod
    def forward(ctx, z, spect, speaker_ids, F, *F_weights):
        ctx.F = F
        with torch.no_grad():
            audio_0, audio_1 = z.chunk(2, 1)
            audio_0, audio_1 = audio_0.contiguous(), audio_1.contiguous()
        
            log_s, t = F(audio_0, spect, speaker_ids)
            audio_1_out = audio_1 * log_s.exp() + t
            audio_0_out = audio_0
            audio_out = torch.cat((audio_0_out, audio_1_out), 1)
        
        ctx.save_for_backward(z.data, spect, speaker_ids, audio_out)
        return audio_out, log_s

    @staticmethod
    def backward(ctx, z_grad, log_s_grad):
        F = ctx.F
        z, spect, speaker_ids, audio_out = ctx.saved_tensors
        
        audio_0_out, audio_1_out = audio_out.chunk(2, 1)
        audio_0_out, audio_1_out = audio_0_out.contiguous(), audio_1_out.contiguous()
        dza, dzb = z_grad.chunk(2, 1)
        dza, dzb = dza.contiguous(), dzb.contiguous()
        
        with set_grad_enabled(True):
            audio_0 = audio_0_out
            audio_0.requires_grad = True
            log_s, t = F(audio_0, spect, speaker_ids)
        
        with torch.no_grad():
            s = log_s.exp()
            audio_1 = (audio_1_out - t) / s
            z.storage().resize_(reduce(mul, audio_1.shape) * 2)
            torch.cat((audio_0, audio_1), 1, out=z) #fp32  # .contiguous()
            #torch.cat((audio_0.half(), audio_1.half()), 1, out=z)#fp16  # .contiguous()
            #z.copy_(xout)  # .detach()
        
        with set_grad_enabled(True):
            param_list = [audio_0] + list(F.parameters())
            if ctx.needs_input_grad[1]:
                param_list += [spect, speaker_ids]
            dtsdxa, *dw = grad(torch.cat((log_s, t), 1), param_list,
                               grad_outputs=torch.cat((dzb * audio_1 * s + log_s_grad, dzb), 1))
            
            dxa = dza + dtsdxa
            dxb = dzb * s
            dx = torch.cat((dxa, dxb), 1)
            if ctx.needs_input_grad[1]:
                *dw, dy = dw
            else:
                dy = None
            if ctx.needs_input_grad[2]:
                *dw, ds = dw
            else:
                ds = None
        
        return (dx, dy, ds, None) + tuple(dw)


class InvAffineCouplingFunc(Function):
    @staticmethod
    def forward(ctx, audio_out, spect, speaker_ids, F, *F_weights):
        ctx.F = F
        with torch.no_grad():
            audio_0_out, audio_1_out = audio_out.chunk(2, 1)
            audio_0_out, audio_1_out = audio_0_out.contiguous(), audio_1_out.contiguous()
            
            log_s, t = F(audio_0_out, spect, speaker_ids)
            audio_1 = (audio_1_out - t) / log_s.exp()
            audio_0 = audio_0_out
            z = torch.cat((audio_0, audio_1), 1)
        
        ctx.save_for_backward(audio_out.data, spect, speaker_ids, z)
        return z, -log_s
    
    @staticmethod
    def backward(ctx, x_grad, log_s_grad):
        F = ctx.F
        audio_out, spect, speaker_ids, z = ctx.saved_tensors
        
        audio_0, audio_1 = z.chunk(2, 1)
        audio_0, audio_1 = audio_0.contiguous(), audio_1.contiguous()
        dxa, dxb = x_grad.chunk(2, 1)
        dxa, dxb = dxa.contiguous(), dxb.contiguous()
        
        with set_grad_enabled(True):
            audio_0_out = audio_0
            audio_0_out.requires_grad = True
            log_s, t = F(audio_0_out, spect, speaker_ids)
            s = log_s.exp()
        
        with torch.no_grad():
            audio_1_out = audio_1 * s + t
            
            audio_out.storage().resize_(reduce(mul, audio_1_out.shape) * 2)
            torch.cat((audio_0_out, audio_1_out), 1, out=audio_out)
            #audio_out.copy_(zout)
        
        with set_grad_enabled(True):
            param_list = [audio_0_out] + list(F.parameters())
            if ctx.needs_input_grad[1]:
                param_list += [spect]
            if ctx.needs_input_grad[2]:
                param_list += [speaker_ids]
            dtsdza, *dw = grad(torch.cat((-log_s, -t / s), 1), param_list,
                               grad_outputs=torch.cat((dxb * audio_1_out / s.detach() + log_s_grad, dxb), 1))
            
            dza = dxa + dtsdza
            dzb = dxb / s.detach()
            dz = torch.cat((dza, dzb), 1)
            if ctx.needs_input_grad[1]:
                *dw, dy = dw
            else:
                dy = None
            if ctx.needs_input_grad[2]:
                *dw, ds = dw
            else:
                ds = None
            
        return (dz, dy, ds, None) + tuple(dw)