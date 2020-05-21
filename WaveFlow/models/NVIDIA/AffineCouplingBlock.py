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


class AffineCouplingBlock(nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """
    def __init__(self, transform_type, hparams):
        super(AffineCouplingBlock, self).__init__()
        if hparams.ACB_memory_efficient:
			raise NotImplementedError # memory_efficient hasn't been implemented for Nvidia based AffineCouplingBlock
		
        self.WN = transform_type(hparams)
    
    def forward(self, audio, spect, speaker_ids):
		audio_0, audio_1 = audio.chunk(2,1)
		b, log_s = self.WN(audio_0, spect, speaker_ids=speaker_ids)
		audio_1 = torch.exp(log_s)*audio_1 + b
		audio = torch.cat([audio_0, audio_1],1)
		return audio, log_s
    
    def inverse(self, audio, spect, speaker_ids):
		n_half = int(audio.size(1)/2)
		audio_0 = audio[:,:n_half,:]
		audio_1 = audio[:,n_half:,:]
		b, s = self.WN[k](audio_0, spect, speaker_ids=speaker_ids)
		audio_1 = (audio_1 - b)/torch.exp(s)
		audio = torch.cat([audio_0, audio_1],1)
		return audio