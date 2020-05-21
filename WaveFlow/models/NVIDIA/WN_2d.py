# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a+input_b
    t_act =    torch.tanh(in_act[:, :n_channels_int, :, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :, :])
    acts = t_act * s_act
    return acts


class WN(nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary difference
    from WaveNet is the convolutions need not be causal.  There is also no dilation
    size reset.  The dilation only doubles on each layer
    """
    def __init__(self, hparams): # bool: ReZero
        super(WN, self).__init__()
        assert(hparams.kernel_width % 2 == 1)
        assert(hparams.n_channels % 2 == 0)
        self.n_layers = hparams.n_layers
        self.n_channels = hparams.n_channels
        self.speaker_embed_dim = hparams.speaker_embed_dim
        
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        if hparams.rezero:
            self.alpha_i = nn.ParameterList()
        
        if self.speaker_embed_dim:
            self.speaker_embed = nn.Embedding(hparams.max_speakers, self.speaker_embed_dim)
        
        n_in_channels = hparams.waveglow_n_half
        
        start = nn.Conv2d(1, self.n_channels,(1, 1))
        start = nn.utils.weight_norm(start, name='weight')
        self.start = start
        
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = nn.Conv2d(self.n_channels, 2, (1, 1))
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end
        
        cond_layer = nn.Conv2d(hparams.n_mel_channels+self.speaker_embed_dim, 2*self.n_channels*self.n_layers, (1, 1))# (in_channels, out_channels, kernel_size)
        self.cond_layer = nn.utils.weight_norm(cond_layer, name='weight')
        
        for i in range(self.n_layers):
            dilation_w = 2 ** i
            dilation_h = 1
            padding_w = int((hparams.kernel_width*dilation_w - dilation_w)/2)
            padding_h = int((hparams.kernel_height*dilation_h - dilation_h)/2)
            in_layer = nn.Conv2d(self.n_channels, 2*self.n_channels, (hparams.kernel_height, hparams.kernel_width),
                                       dilation=(dilation_h, dilation_w), padding=(padding_h, padding_w))
            in_layer = nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)
            
            # last one is not necessary
            if i < self.n_layers - 1:
                res_skip_channels = 2*self.n_channels
            else:
                res_skip_channels = self.n_channels
            res_skip_layer = nn.Conv2d(self.n_channels, res_skip_channels, (1, 1))
            res_skip_layer = nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)
            
            if hparams.rezero:
                alpha_ = nn.Parameter(torch.rand(1)*0.02+0.09) # rezero initial state (0.1±0.01)
                self.alpha_i.append(alpha_)
    
    def forward(self, audio, spect, speaker_ids=None):
        audio = self.start(audio) # [B, 1, n_group//2, T//n_group]
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])
        
        if self.speaker_embed_dim and speaker_ids != None:
            speaker_embeddings = self.speaker_embed(speaker_ids) # [B, speaker_embed_dim]
            speaker_embeddings = speaker_embeddings.view(*speaker_embeddings.shape,1,1).repeat(1, 1, spect.shape[2],spect.shape[3]) # shape like spect [B, speaker_embed_dim] -> [B, speaker_embed_dim, 1, 1] -> [B, speaker_embed_dim, n_group, T//n_group]
            spect = torch.cat([spect, speaker_embeddings], dim=1) # and concat them
        
        spect = self.cond_layer(spect) # [B, n_mel, n_group, T//n_group]
        
        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels, (i+1)*2*self.n_channels
            acts = fused_add_tanh_sigmoid_multiply(
                self.in_layers[i](audio),
                spect[:,spect_offset[0]:spect_offset[1],:,:],
                n_channels_tensor)
            
            if hasattr(self, 'alpha_i'): # if rezero
                res_skip_acts = self.res_skip_layers[i](acts) * self.alpha_i[i]
            else:
                res_skip_acts = self.res_skip_layers[i](acts)
            
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:,:self.n_channels,:]
                output = output + res_skip_acts[:,self.n_channels:,:]
            else:
                output = output + res_skip_acts
        
        return self.end(output).chunk(2, 1)