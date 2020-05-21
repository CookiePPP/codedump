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
from models.model_utils import conditionalImport

class WaveFlow(nn.Module):
    def __init__(self, hparams):
        super(WaveFlow, self).__init__()
        assert(hparams.n_group % 2 == 0)
        assert(not hparams.InvConv_memory_efficient)
        assert(not hparams.ACB_memory_efficient)
        # 'Import' model modules.
        Invertible1x1Conv, AffineCouplingBlock, WN, Upsampler, Squeezer = conditionalImport(hparams)		
        self.upsample = Upsampler(hparams)
        self.squeeze = Squeezer(hparams)
        
        self.multispeaker = hparams.speaker_embed_dim > 0
        self.n_flows = hparams.n_flows
        self.n_group = hparams.n_group
        self.n_early_every = hparams.n_early_every
        self.n_early_size = hparams.n_early_size
        self.WN = nn.ModuleList()
        self.convinv = nn.ModuleList()
        n_half = int(self.n_group/2)
        
        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = self.n_group
        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size/2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            self.convinv.append(Invertible1x1Conv(n_remaining_channels, hparams))
            hparams.waveglow_n_half = n_half
            self.WN.append(AffineCouplingBlock(WN, hparams))
        self.n_remaining_channels = n_remaining_channels  # Useful during inference
	
    def forward(self, spect, audio, speaker_ids=None):
        """
        forward_input[0] = mel_spectrogram:  batch x n_mel_channels x frames
        forward_input[1] = audio: batch x time
        """
        #  Upsample spectrogram to size of audio
        spect = self.upsample(spect)
        audio, spect = self.squeeze(audio, spect)
		
        output_audio = []
        log_s_list = []
        log_det_W_list = []
        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_audio.append(audio[:,:self.n_early_size,:])
                audio = audio[:,self.n_early_size:,:]#.clone() # memory efficient errors.
            
            audio, log_det_W = self.convinv[k](audio)
            log_det_W_list.append(log_det_W)
            
            audio, log_s = self.WN[k](audio, spect, speaker_ids=speaker_ids)
            log_s_list.append(log_s)
        
        output_audio.append(audio)
        return torch.cat(output_audio,1), log_s_list, log_det_W_list

    def infer(self, spect, speaker_ids=None, sigma=1.0):
        spect = self.upsample(spect)
        
        # trim conv artifacts. maybe pad spec to kernel multiple
        try:
            time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
            spect = spect[:, :, :-time_cutoff]
        except:
            pass
        
        spect = self.squeeze(None, spect)[1]
        
        audio = torch.ones(spect.size(0), self.n_remaining_channels, spect.size(2), device=spect.device, dtype=spect.dtype).normal_(std=sigma)
        
        for k in reversed(range(self.n_flows)):
            audio, *_ = self.WN[k](audio, spect, speaker_ids=speaker_ids)
            audio = self.convinv[k](audio, reverse=True)
            
            if k % self.n_early_every == 0 and k > 0:
                z = torch.ones(spect.size(0), self.n_early_size, spect.size(2), device=spect.device, dtype=spect.dtype).normal_(std=sigma)
                audio = torch.cat((z, audio),1)
        
        audio = audio.permute(0,2,1).contiguous().view(audio.size(0), -1).data
        return audio
    
    @staticmethod
    def remove_weightnorm(model):
        WaveFlow = model
        for WN in WaveFlow.WN:
            WN.start = nn.utils.remove_weight_norm(WN.start)
            WN.in_layers = remove_norms(WN.in_layers)
            WN.cond_layer = nn.utils.remove_weight_norm(WN.cond_layer)
            WN.res_skip_layers = remove_norms(WN.res_skip_layers)
        return WaveFlow


def remove_norms(conv_list):
    new_conv_list = nn.ModuleList()
    for old_conv in conv_list:
        old_conv = nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
    return new_conv_list