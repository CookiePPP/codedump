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
		
		if audio is not None:  # during synthesize phase, we feed audio as None
			# squeeze 1D waveform audio into 2d matrix given height self.n_group
			audio = audio.unsqueeze(1) # [B, T] -> [B, 1, T] # not sure what dim=1 does/is used for
			
			B, C, T = audio.size()
			assert T % self.n_group == 0, "cannot make 2D matrix of size {} given self.n_group={}".format(T, self.n_group)
			audio = audio.view(B, int(T / self.n_group), C * self.n_group) # [B, 1, T] -> [B, T//n_group, n_group]
			# permute to make column-major 2D matrix of waveform
			
			audio = audio.permute(0, 2, 1) # [B, T//n_group, n_group] -> [B, n_group, T//n_group]
			
			audio = audio.unsqueeze(1) # [B, n_group, T//n_group] -> [B, 1, n_group, T//n_group]
		
		# same goes to mel, but keeping the 2D mel-spec shape
		B, C, T = mel.size()
		mel = mel.view(B, C, int(T / self.n_group), self.n_group) # [B, n_mel, T] -> [B, n_mel, T//n_group, n_group]
		mel = mel.permute(0, 1, 3, 2) # [B, n_mel, T//n_group, n_group] -> [B, n_mel, n_group, T//n_group]
		
		return audio, mel # [B, 1, n_group, T//n_group], [B, n_mel, n_group, T//n_group]