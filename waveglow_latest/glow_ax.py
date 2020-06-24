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
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class WaveGlowLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super(WaveGlowLoss, self).__init__()
        self.sigma = sigma
        self.sigma2 = sigma*sigma
        self.sigma2_2 = 2*sigma*sigma
    
    def forward(self, model_output):
        z, log_s_list, log_det_W_list = model_output
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_W_total += log_det_W_list[i]
        
        loss = torch.sum(z*z)/(self.sigma2_2) - log_s_total - log_det_W_total
        return loss/(z.size(0)*z.size(1)*z.size(2))


class Invertible1x1Conv(nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """
    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                    bias=False)
        
        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]
        
        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:,0] = -1*W[:,0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W
    
    def forward(self, z, reverse=False):
        # shape
        batch_size, group_size, n_of_groups = z.size()
        W = self.conv.weight.squeeze()
        
        if reverse:
            if not hasattr(self, 'W_inverse'):
                # Reverse computation
                W_inverse = W.float().inverse()
                W_inverse = Variable(W_inverse[..., None])
                if z.dtype == 'torch.float16':
                    W_inverse = W_inverse.half()
                self.W_inverse = W_inverse
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            # Forward computation
            if z.dtype == 'torch.float16':
                log_det_W = batch_size * n_of_groups * torch.logdet(W.float()).half()
            else:
                log_det_W = batch_size * n_of_groups * torch.logdet(W)
            z = self.conv(z)
            return z, log_det_W


class WN(nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary difference
    from WaveNet is the convolutions need not be causal.  There is also no dilation
    size reset.  The dilation only doubles on each layer
    """
    def __init__(self, n_in_channels, cond_in_channels, cond_layers, cond_hidden_channels, cond_kernel_size, cond_padding_mode, seperable_conv, merge_res_skip, upsample_mode, n_layers, n_channels, # audio_channels, mel_channels*n_group, n_layers, n_conv_channels
                 kernel_size_w, kernel_size_h, speaker_embed_dim, rezero, cond_activation_func='none', negative_slope=None, n_layers_dilations_w=None, n_layers_dilations_h=1, res_skip=True):
        super(WN, self).__init__()
        assert(kernel_size_w % 2 == 1)
        assert(n_channels % 2 == 0)
        assert res_skip or merge_res_skip, "Cannot remove res_skip without using merge_res_skip"
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.kernel_size_h = kernel_size_h
        self.speaker_embed_dim = speaker_embed_dim
        self.merge_res_skip = merge_res_skip
        self.upsample_mode = upsample_mode
        
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        
        assert (not rezero), "WN ReZero is depreciated"
        if rezero:
            self.alpha_i = nn.ParameterList()
        
        start = nn.Conv2d(1, n_channels, (1,1))
        start = nn.utils.weight_norm(start, name='weight')
        self.start = start
        
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = nn.Conv2d(n_channels, 2, (1,1))
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end
        
        if self.speaker_embed_dim:
            max_speakers = 512
            self.speaker_embed = nn.Embedding(max_speakers, self.speaker_embed_dim)
        
        self.cond_layers = nn.ModuleList()
        if cond_layers:
            cond_in_channels = cond_in_channels + self.speaker_embed_dim
            cond_kernel_size = 2*cond_kernel_size - 1 # 1 -> 1, 2 -> 3, 3 -> 5
            cond_pad = int((cond_kernel_size - 1)/2)
            cond_output_channels = 2*n_channels*n_layers
            # messy initialization for arbitrary number of layers, input dims and output dims
            dimensions = [cond_in_channels,]+[cond_hidden_channels]*(cond_layers-1)+[cond_output_channels,]
            in_dims = dimensions[:-1]
            out_dims = dimensions[1:]
            # 'zeros','replicate'
            for i in range(len(in_dims)):
                indim = in_dims[i]
                outim = out_dims[i]
                cond_layer = nn.Conv1d(indim, outim, cond_kernel_size, padding=cond_pad, padding_mode=cond_padding_mode)# (in_channels, out_channels, kernel_size)
                cond_layer = nn.utils.weight_norm(cond_layer, name='weight')
                self.cond_layers.append(cond_layer)
            
            cond_activation_func = cond_activation_func.lower()
            if cond_activation_func == 'none':
                pass
            elif cond_activation_func == 'lrelu':
                self.cond_activation_func = torch.nn.functional.relu
            elif cond_activation_func == 'relu':
                assert negative_slope, "negative_slope not defined in wn_config"
                self.cond_activation_func = torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
            elif cond_activation_func == 'tanh':
                self.cond_activation_func = torch.nn.functional.tanh
            elif cond_activation_func == 'sigmoid':
                self.cond_activation_func = torch.nn.functional.sigmoid
            else:
                raise NotImplementedError
        
        if type(n_layers_dilations_w) == int:
            n_layers_dilations_w = [n_layers_dilations_w,]*n_layers # constant dilation if using int
            print("WARNING: Using constant dilation factor for WN in_layer dilation width.")
        if type(n_layers_dilations_h) == int:
            n_layers_dilations_h = [n_layers_dilations_h,]*n_layers # constant dilation if using int
        
        self.h_dilate = n_layers_dilations_h
        self.padding_h = []
        for i in range(n_layers):
            dilation_h = n_layers_dilations_h[i]
            dilation_w = 2 ** i if n_layers_dilations_w is None else n_layers_dilations_w[i]
            
            padding_w = ((kernel_size_w-1)*dilation_w)//2
            self.padding_h.append((kernel_size_h-1)*dilation_h) # causal padding https://theblog.github.io/post/convolution-in-autoregressive-neural-networks/
            if (not seperable_conv) or (kernel_size_w == 1 and kernel_size_h == 1):
                in_layer = nn.Conv2d(n_channels, 2*n_channels, (kernel_size_h,kernel_size_w),
                                           dilation=(dilation_h,dilation_w), padding=(0,padding_w), padding_mode='zeros')
                in_layer = nn.utils.weight_norm(in_layer, name='weight')
            else:
                depthwise = nn.Conv2d(n_channels, n_channels, (kernel_size_h,kernel_size_w),
                                    dilation=(dilation_h,dilation_w), padding=(0,padding_w), padding_mode='zeros', groups=n_channels)
                depthwise = nn.utils.weight_norm(depthwise, name='weight')
                pointwise = nn.Conv2d(n_channels, 2*n_channels, (1,1),
                                    dilation=(1,1), padding=(0,0))
                pointwise = nn.utils.weight_norm(pointwise, name='weight')
                in_layer = torch.nn.Sequential(depthwise, pointwise)
            self.in_layers.append(in_layer)
            
            # last one is not necessary
            if i < n_layers - 1 and not self.merge_res_skip:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            
            if res_skip:
                res_skip_layer = nn.Conv2d(n_channels, res_skip_channels, (1,1))
                res_skip_layer = nn.utils.weight_norm(res_skip_layer, name='weight')
                if rezero:
                    alpha_ = nn.Parameter(torch.rand(1)*0.02+0.09) # rezero initial state (0.1±0.01)
                    self.alpha_i.append(alpha_)
                self.res_skip_layers.append(res_skip_layer)
            #
    
    def _upsample_mels(self, cond, audio_size):
        cond = F.interpolate(cond, size=audio_size[3], mode=self.upsample_mode, align_corners=True if self.upsample_mode == 'linear' else None)
        #cond = F.interpolate(cond, scale_factor=600/24, mode=self.upsample_mode, align_corners=True if self.upsample_mode == 'linear' else None) # upsample by hop_length//n_group
        return cond
    
    def forward(self, audio, spect, speaker_id=None, audio_queues=None, spect_queues=None):
        audio = audio.unsqueeze(1) #   [B, n_group//2, T//n_group] -> [B, 1, n_group//2, T//n_group]
        audio = self.start(audio) # [B, 1, n_group//2, T//n_group] -> [B, n_channels, n_group//2, T//n_group]
        if not self.merge_res_skip:
            output = torch.zeros_like(audio) # output and audio are seperate Tensors
        n_channels_tensor = torch.IntTensor([self.n_channels])
        
        if (spect_queues is None) or ( any([x is None for x in spect_queues]) ): # process spectrograms
            if self.speaker_embed_dim and speaker_id != None: # add speaker embeddings to spectrogram (channel dim)
                speaker_embeddings = self.speaker_embed(speaker_id)
                speaker_embeddings = speaker_embeddings.unsqueeze(-1).repeat(1, 1, spect.shape[2]) # shape like spect
                spect = torch.cat([spect, speaker_embeddings], dim=1) # and concat them
            
            for layer in self.cond_layers: # [B, cond_channels, T//hop_length] -> [B, n_channels*n_layers, T//hop_length]
                spect = layer(spect)
                if hasattr(self, 'cond_activation_func'):
                    spect = self.cond_activation_func(spect)
            
            if audio.size(3) > spect.size(2): # if spectrogram hasn't been upsampled yet
                spect = self._upsample_mels(spect, audio.shape)# [B, n_channels*n_layers, T//hop_length] -> [B, n_channels*n_layers, T//n_group]
                spect = spect.unsqueeze(2)# [B, n_channels*n_layers, T//n_group] -> [B, n_channels*n_layers, 1, T//n_group]
                assert audio.size(3) == spect.size(3), f"audio size of {audio.size(3)} != spect size of {spect.size(3)}"
        
        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels, (i+1)*2*self.n_channels
            spec = spect[:,spect_offset[0]:spect_offset[1]] # [B, 2*n_channels*n_layers, 1, T//n_group] -> [B, 2*n_channels, 1, T//n_group]
            
            if spect_queues is not None: # is spect_queues exists...
                if spect_queues[i] is None: # but this index is empty...
                    spect_queues[i] = spec # save spec into this index.
                else:                       # else...
                    spec = spect_queues[i] # load spec from this index.
            
            if audio_queues is None:# if training/validation...
                audio_cpad = F.pad(audio, (0,0,self.padding_h[i],0)) # apply causal height padding (left, right, top, bottom)
            else: # else, if conv-queue and inference/autoregressive sampling.
                if audio_queues[i] is None: # if first sample in autoregressive sequence, pad start with zeros
                    B, n_channels, n_group, T_group = audio.shape
                    audio_queues[i] = audio.new_zeros( size=[B, n_channels, self.padding_h[i], T_group] )
                
                # [B, n_channels, n_group, T//n_group]
                audio_queues[i] = audio_cpad = torch.cat((audio_queues[i], audio), dim=2)[:,:,-(self.padding_h[i]+1):] # pop old samples and append new sample to end of n_group dim
                assert audio_cpad.shape[2] == (self.padding_h[i]+self.h_dilate[i]), f"conv queue is wrong length. Found {audio_cpad.shape[2]}, expected {(self.padding_h[i]+self.h_dilate[i])}"
            
            acts = self.in_layers[i](audio_cpad) # [B, n_channels, n_group//2, T//n_group] -> [B, 2*n_channels, pad+n_group//2, T//n_group]
            acts = fused_add_tanh_sigmoid_multiply(
                acts, # [B, 2*n_channels, n_group//2, T//n_group]
                spec, # [B, 2*n_channels, 1, T//n_group]
                n_channels_tensor)
            # acts.shape <- [B, n_channels, n_group//2, T//n_group]
            
            if hasattr(self, 'res_skip_layers') and len(self.res_skip_layers):
                if hasattr(self, 'alpha_i'): # if rezero
                    res_skip_acts = self.res_skip_layers[i](acts) * self.alpha_i[i]
                else:
                    res_skip_acts = self.res_skip_layers[i](acts)
            else:
                res_skip_acts = acts
                # if merge_res_skip: [B, n_channels, n_group//2, T//n_group] -> [B, n_channels, n_group//2, T//n_group]
                # else: [B, n_channels, n_group//2, T//n_group] -> [B, 2*n_channels, n_group//2, T//n_group]
            
            if self.merge_res_skip:
                audio = audio + res_skip_acts
            else:
                if i < self.n_layers - 1:
                    audio = audio + res_skip_acts[:,:self.n_channels,:]
                    output = output + res_skip_acts[:,self.n_channels:,:]
                else:
                    output = output + res_skip_acts
        
        if self.merge_res_skip:
            output = audio
        
        func_out = self.end(output).transpose(1,0) # [B, n_channels, n_group//2, T//n_group] -> [B, 2, n_group//2, T//n_group] -> [2, B, n_group//2, T//n_group]
        
        if audio_queues is not None:
            func_out = [func_out,]
            func_out.append(audio_queues)
        if spect_queues is not None:
            func_out.append(spect_queues)
        return func_out


class WaveGlow(nn.Module):
    def __init__(self, yoyo, yoyo_WN, n_mel_channels, n_flows, n_group, n_early_every,
                 n_early_size, memory_efficient, spect_scaling, upsample_mode, WN_config, win_length, hop_length):
        super(WaveGlow, self).__init__()
        assert not spect_scaling, "spect_scaling is depreciated."
        self.multispeaker = WN_config['speaker_embed_dim'] > 0
        
        #upsample_mode = 'normal' # options: 'normal','simple','simple_half'
        self.upsample = nn.ConvTranspose1d(n_mel_channels,
                                                 n_mel_channels,
                                                 win_length, stride=hop_length,
                                                 groups=1 if upsample_mode == 'normal' else (n_mel_channels if upsample_mode == 'simple' else (n_mel_channels/2 if upsample_mode == 'simple_half' else print("upsample_mode = {upsample_mode} invalid"))) )
        
        assert(n_group % 2 == 0)
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.WN = nn.ModuleList()
        self.convinv = nn.ModuleList()
        
        n_half = int(n_group/2)
        
        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = n_group
        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size/2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            if not memory_efficient: # normal
                self.convinv.append(Invertible1x1Conv(n_remaining_channels))
                self.WN.append(WN(n_half, n_mel_channels*n_group, **WN_config))
            else: # mem_efficient
                pass
        self.n_remaining_channels = n_remaining_channels  # Useful during inference

    def forward(self, spect, audio, speaker_id=None):
        """
        forward_input[0] = mel_spectrogram:  batch x n_mel_channels x frames
        forward_input[1] = audio: batch x time
        """
        if self.spect_scaling:
            spect.mul_(self.spect_scale).add_(self.spect_shift) # adjust each spectogram channel by a param
        
        #  Upsample spectrogram to size of audio
        spect = self.upsample(spect)
        assert(spect.size(2) >= audio.size(1))
        if spect.size(2) > audio.size(1):
            spect = spect[:, :, :audio.size(1)]
        
        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)
        
        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1) # "Squeeze to Vectors"
        output_audio = []
        log_s_list = []
        log_det_W_list = []
        
        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_audio.append(audio[:,:self.n_early_size,:])
                audio = audio[:,self.n_early_size:,:]#.clone() # memory efficient errors.
            
            audio, log_det_W = self.convinv[k](audio)
            log_det_W_list.append(log_det_W)
            
            audio_0, audio_1 = audio.chunk(2,1)
            #n_half = int(audio.size(1)/2)
            #audio_0 = audio[:,:n_half,:]
            #audio_1 = audio[:,n_half:,:]
            
            #output = self.WN[k]((audio_0, spect))
            #log_s = output[:, n_half:, :]
            #b = output[:, :n_half, :]
            b, log_s = self.WN[k](audio_0, spect, speaker_id=speaker_id)
            audio_1 = torch.exp(log_s)*audio_1 + b
            log_s_list.append(log_s)
            
            audio = torch.cat([audio_0, audio_1],1)
        
        output_audio.append(audio)
        return torch.cat(output_audio,1), log_s_list, log_det_W_list

    def infer(self, spect, speaker_id=None, sigma=1.0):
        if self.spect_scaling:
            spect.mul_(self.spect_scale).add_(self.spect_shift) # adjust each spectogram channel by a param

        spect = self.upsample(spect)
        # trim conv artifacts. maybe pad spec to kernel multiple
        time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
        spect = spect[:, :, :-time_cutoff]
        
        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)
        
        audio = torch.ones(spect.size(0), self.n_remaining_channels, spect.size(2), device=spect.device, dtype=spect.dtype).normal_(std=sigma)
        
        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1)/2)
            audio_0 = audio[:,:n_half,:]
            audio_1 = audio[:,n_half:,:]
            
            b, s = self.WN[k](audio_0, spect, speaker_id=speaker_id)
            
            #s = output[:, n_half:, :]
            #b = output[:, :n_half, :]
            audio_1 = (audio_1 - b)/torch.exp(s)
            audio = torch.cat([audio_0, audio_1],1)
            
            audio = self.convinv[k](audio, reverse=True)
            
            if k % self.n_early_every == 0 and k > 0:
                if spect.type() == 'torch.cuda.HalfTensor':
                    z = torch.cuda.HalfTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()
                else:
                    z = torch.cuda.FloatTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()
                audio = torch.cat((sigma*z, audio),1)
        
        audio = audio.permute(0,2,1).contiguous().view(audio.size(0), -1).data
        return audio

    @staticmethod
    def remove_weightnorm(model):
        waveglow = model
        for WN in waveglow.WN:
            WN.start = nn.utils.remove_weight_norm(WN.start)
            WN.in_layers = remove(WN.in_layers)
            WN.cond_layer = nn.utils.remove_weight_norm(WN.cond_layer)
            WN.res_skip_layers = remove(WN.res_skip_layers)
        return waveglow


def remove(conv_list):
    new_conv_list = nn.ModuleList()
    for old_conv in conv_list:
        old_conv = nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
    return new_conv_list
