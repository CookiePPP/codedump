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
        
    def forward(self, audio, mel):
        # audio [B, T]
        # mel [B, n_mel, T//n_group]

        if audio is not None:
            assert mel.shape[2] >= audio.shape[1]
            # Prune out the tail of audio/mel so that T//n_group == 0.
            pruned_len = audio.shape[1] // self.n_group * self.n_group
            
            if audio.shape[1] > pruned_len:
                audio = audio[:, :pruned_len]
            if mel.shape[2] > pruned_len:
                mel = mel[:, :, :pruned_len]
            
            # From [B, T] to [B, n_group, T//n_group]
            audio = audio.unfold(1, self.n_group, self.n_group).transpose(2, 1)
            
            # [B, 1, n_group, T//n_group] 
            audio = audio.unsqueeze(1)
        
        # From [B, n_mel, T] to [B, n_mel, n_group, T//n_group]
        mel = mel.unfold(2, self.n_group, self.n_group).transpose(3, 2)
        
        return audio, mel # [B, 1, n_group, T//n_group], [B, n_mel, n_group, T//n_group]