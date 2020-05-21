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
        # mel [B, n_mel, T//n_group]

        assert(spect.size(2) >= audio.size(1))
        if spect.size(2) > audio.size(1):
            spect = spect[:, :, :audio.size(1)]

        # spect.shape = torch.Size([5, 160, 6000])
        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3) # [B, n_mels, T] -> [B, T//n_group, n_mels, n_group]
        # spect.shape = torch.Size([5, 750, 160, 8])
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1) # [B, n_mels*n_group, T//n_group]
        # spect.shape = torch.Size([5, 1280, 750])

        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1) # [B, T] -> [B, n_group, T//n_group]
        # audio.shape = torch.Size([5, 8, 750])

        return audio, spect # [B, n_group, T//n_group], [B, n_mels*n_group, T//n_group]