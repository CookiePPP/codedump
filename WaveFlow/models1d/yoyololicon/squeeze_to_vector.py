# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ~~~~~~~~~~~~~ COOKIE NOTE ~~~~~~~~~~~~~
# Modifications made to the original code are the following;
# - change all appropriate functions from their PaddlePaddle to Pytorch Equivalents
# - package audio squeeze into single function


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Squeezer(nn.Module):
    """
    Squeeze audio and mel-spectrogram into vectors.
    """
    def __init__(self, hparams):
        super(Squeezer, self).__init__()
        self.n_group = hparams.n_group
        
    def forward(self, audio, spect):
        # audio [B, T]
        # spect [B, n_mel, T//n_group]
        
        batch_dim, n_mel_channels, group_steps = spect.shape
        audio = audio.view(batch_dim, -1, self.n_group).transpose(1, 2) # [B, T] -> [B, n_group, T//n_group]
        
        assert audio.size(2) <= spect.size(2)
        spect = spect[..., :audio.size(2)]

        return audio, spect # [B, n_group, T//n_group], [B, n_mel, T//n_group]