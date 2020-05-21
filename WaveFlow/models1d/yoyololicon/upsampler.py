import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Upsampler(nn.Module):
    """
    Upsamples 2nd dim by self.upsample_factor using nearest-neighbour upsampling (dumb/fast method).
    eg:
    if input has shape [1,160,10] and
    self.upsample_factors = 5
    then output will have shape [1,160,50] where it was upsamples by a factor of 5.
    """
    def __init__(self, hparams):
        super(Upsampler, self).__init__()
        self.hop_length = hparams.hop_length
        #self.upsample_factor = self.hop_length // self.n_group # outputs [B, n_mels, T//n_group]
        self.upsample_factor = self.hop_length					# outputs [B, n_mels, T]
        
    def _upsample_mels(self, spect):
        spect = F.pad(spect, (0, 1))
        return F.interpolate(spect, size=((spect.size(2) - 1) * self.upsample_factor + 1,), mode='linear')

    def forward(self, spect):
        # [B, n_mels, T//hop_length]

        spect = self._upsample_mels(spect) # [B, n_mels, T//hop_length] -> [B, n_mels, T]

        return spect #[B, n_mels, T]