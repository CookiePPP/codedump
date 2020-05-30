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
                 radix=3, bias=False, speaker_embed_dim=0, zero_init=True):
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
    
    def forward(self, audio, spect, speaker_ids):
        audio = self.start(audio)
        
        cum_skip = None
        for layer in self.layers:
            audio, skip = layer(audio, spect)
            if cum_skip is None:
                cum_skip = skip
            else:
                cum_skip += skip
        return self.end(cum_skip).chunk(2, 1)


class AffineCouplingBlock(nn.Module):
    def __init__(self,
                 transform_type,
                 memory_efficient=True,
                 **kwargs):
        super().__init__()
        
        self.WN = transform_type(**kwargs)
        if memory_efficient:
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
            ###print("1 audio_0.dtype =", audio_0.dtype)
            ###print("1 audio_1.dtype =", audio_1.dtype)
            ###print("1 audio_0_out.dtype =", audio_0_out.dtype)
            ###print("1 audio_1_out.dtype =", audio_1_out.dtype)
            ###print("1 log_s.dtype =", log_s.dtype)
            ###print("1 t.dtype =", t.dtype)
        
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
            ###print("2 audio_0_out.dtype =", audio_0_out.dtype)
            ###print("2 audio_1_out.dtype =", audio_1_out.dtype)
            ###print("2 log_s.dtype =", log_s.dtype)
            ###print("2 t.dtype =", t.dtype)
        
        with torch.no_grad():
            s = torch.exp(log_s).half() # exp not implemented for fp16 therefore this is cast to fp32 by Nvidia/Apex
            audio_1 = (audio_1_out - t) / s # s is fp32 therefore audio_1 is cast to fp32.
            z.storage().resize_(reduce(mul, audio_1.shape) * 2) # z is fp16
            ###print("3 log_s.dtype =", log_s.dtype)
            ###print("3 s.dtype =", s.dtype)
            ###print("3 t.dtype =", t.dtype)
            ###print("3 audio_0.dtype =", audio_0.dtype)
            ###print("3 audio_1.dtype =", audio_1.dtype)
            ###print("3 z.dtype =", z.dtype)
            if z.dtype == torch.float16: # if z is fp16, cast audio_0 and audio_1 back to fp16.
              torch.cat((audio_0.half(), audio_1.half()), 1, out=z)#fp16  # .contiguous()
            else:
              torch.cat((audio_0, audio_1), 1, out=z) #fp32  # .contiguous()
            #z.copy_(xout)  # .detach()
        
        with set_grad_enabled(True):
            param_list = [audio_0] + list(F.parameters())
            if ctx.needs_input_grad[1]:
                param_list += [spect]
            if ctx.needs_input_grad[2]:
                param_list += [speaker_ids]
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


class InvertibleConv1x1(nn.Conv1d):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """
    def __init__(self, c, memory_efficient=False):
        super().__init__(c, c, 1, bias=False) # init as nn.Conv1d(c, c, kernel_size=1, stride=1) 
        
        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]
        
        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:,0] = -1*W[:,0]
        W = W.view(c, c, 1)
        self.weight.data = W
        
        if memory_efficient:
            self.efficient_forward = Conv1x1Func.apply
            #self.efficient_inverse = InvConv1x1Func.apply # memory efficient Inverse is not needed and it's making fp16 more complicated so I'm ignoring it for now.
    
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
        W = self.weight.squeeze()
        if not hasattr(self, 'W_inverse'):
            W_inverse = W.float().inverse().unsqueeze(-1)
            if audio_out.dtype == torch.float16:
                W_inverse = W_inverse.half()
            self.W_inverse = W_inverse
        
        if hasattr(self, 'efficient_inverse'):
            z, log_det_W = self.efficient_inverse(audio_out, self.weight)
            audio_out.storage().resize_(0)
            return z, log_det_W
        else:
            log_det_W = None
            #*_, n_of_groups = audio_out.shape
            #log_det_W = -n_of_groups * weight.slogdet()[1]  # should fix nan logdet
            z = F.conv1d(audio_out, self.W_inverse, bias=None, stride=1, padding=0)
            return z, log_det_W


class Conv1x1Func(Function):
    @staticmethod
    def forward(ctx, z, weight):
        with torch.no_grad():
            *_, n_of_groups = z.shape
            if weight.dtype == torch.float16:
                log_det_W = n_of_groups * weight.squeeze().float().slogdet()[1].half()
            else:
                log_det_W = n_of_groups * weight.squeeze().slogdet()[1]
            audio_out = F.conv1d(z, weight)
        
        ctx.save_for_backward(z.data, weight, audio_out)
        return audio_out, log_det_W
    
    @staticmethod
    def backward(ctx, z_grad, log_det_W_grad):
        z, weight, audio_out = ctx.saved_tensors
        *_, n_of_groups = audio_out.shape
        
        with torch.no_grad():
            if weight.dtype == torch.float16:
                inv_weight = weight.squeeze().float().inverse().half()
            else:
                inv_weight = weight.squeeze().inverse()
            z.storage().resize_(reduce(mul, audio_out.shape))
            z[:] = F.conv1d(audio_out, inv_weight.unsqueeze(-1))
            
            dx = F.conv1d(z_grad, weight[..., 0].t().unsqueeze(-1))
            dw = z_grad.transpose(0, 1).contiguous().view(weight.shape[0], -1) @ z.transpose(1, 2).contiguous().view(
                -1, weight.shape[1])
            dw += inv_weight.t() * log_det_W_grad * n_of_groups
        
        return dx, dw.unsqueeze(-1)


class InvConv1x1Func(Function):
    @staticmethod
    def forward(ctx, z, inv_weight, weight):
        with torch.no_grad():
            squ_weight = weight.squeeze()
            *_, n_of_groups = z.shape
            if squ_weight.dtype == torch.float16:
              log_det_W = -squ_weight.float().slogdet()[1].half() * n_of_groups
              audio_out = F.conv1d(z, squ_weight.inverse().unsqueeze(-1).half())
            else:
              log_det_W = -squ_weight.slogdet()[1] * n_of_groups
              audio_out = F.conv1d(z, squ_weight.inverse().unsqueeze(-1))
        
        ctx.save_for_backward(z.data, weight, audio_out)
        return audio_out, log_det_W
    
    @staticmethod
    def backward(ctx, z_grad, log_det_W_grad):
        z, weight, audio_out = ctx.saved_tensors
        *_, n_of_groups = audio_out.shape
        
        with torch.no_grad():
            z.storage().resize_(reduce(mul, audio_out.shape))
            z[:] = F.conv1d(audio_out, weight)
            
            weight = weight.squeeze()
            weight_T = weight.inverse().t()
            dx = F.conv1d(z_grad, weight_T.unsqueeze(-1))
            dw = z_grad.transpose(0, 1).contiguous().view(weight_T.shape[0], -1) @ \
                 z.transpose(1, 2).contiguous().view(-1, weight_T.shape[1])
            dinvw = - weight_T @ dw @ weight_T
            dinvw -= weight_T * log_det_W_grad * n_of_groups
        
        return dx, dinvw.unsqueeze(-1)
