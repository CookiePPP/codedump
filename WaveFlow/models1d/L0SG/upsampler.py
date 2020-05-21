import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Upsampler(nn.Module):
	"""
	Upsamples 2nd dim by self.upsample_factors
	eg:
	if input has shape [1,160,10] and
	self.upsample_factors = [10,10]
	then output will have shape [1,160,1000] where it was upsamples by a factor of 10 twice.
	"""
    def __init__(self, hparams):
        super(Upsampler, self).__init__()
		self.upsample_factors = hparams.upsample_factors
		
        self.upsample_conv = nn.ModuleList()
        for s in self.upsample_factors:
            convt = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s), groups=upsample_groups)
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self.upsample_conv.append(convt)
            self.upsample_conv.append(nn.LeakyReLU(0.4))
	
    def forward(self, spect):
        spect = spect.unsqueeze(1)
        for f in self.upsample_conv:
            spect = f(spect)
        spect = spect.squeeze(1)
        return spect