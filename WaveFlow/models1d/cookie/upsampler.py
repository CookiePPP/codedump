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
	
    def forward(self, spect):
		# [B, n_mels, T//hop_length]
		
        spect = spect.repeat(1,1,self.hop_length) # [B, n_mels, T//hop_length] -> [B, n_mels, T]
		
        return spect