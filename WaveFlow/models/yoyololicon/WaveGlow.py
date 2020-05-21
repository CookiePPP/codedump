import torch
import torch.nn as nn
import torch.nn.functional as F
from efficient_util import add_weight_norms
import numpy as np
from models.model_utils import conditionalImport

class WaveGlow(nn.Module):
    def __init__(self, hparams):
#    yoyo, yoyo_WN, n_mel_channels, n_flows, n_group, n_early_every,
#    n_early_size, memory_efficient, spect_scaling, upsample_mode, WN_config, win_length, hop_length):
        super(WaveGlow, self).__init__()
        assert(hparams.n_group % 2 == 0)
        self.n_flows = hparams.n_flows
        self.n_group = hparams.n_group
        self.n_early_every = hparams.n_early_every
        self.n_early_size = hparams.n_early_size
        self.win_size = hparams.win_length
        self.hop_length = hparams.hop_length
        self.n_mel_channels = hparams.n_mel_channels
        self.multispeaker = hparams.speaker_embed_dim > 0
        
        # 'Import' model modules.
        Invertible1x1Conv, AffineCouplingBlock, WN, Upsampler, Squeezer = conditionalImport(hparams)		
        self.upsample = Upsampler(hparams)
        self.squeeze = Squeezer(hparams)
        
        self.convinv = nn.ModuleList()
        self.WN = nn.ModuleList()
        
        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = self.n_group
        self.z_split_sizes = []
        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_remaining_channels -= self.n_early_size
                self.z_split_sizes.append(self.n_early_size)
            
            assert n_remaining_channels > 0 # no n_group remaining
            
            self.convinv.append(
                Invertible1x1Conv(n_remaining_channels, hparams))
            
            hparams.waveglow_n_half = n_remaining_channels//2
            self.WN.append(AffineCouplingBlock(WN, hparams))
        
        self.z_split_sizes.append(n_remaining_channels)
    
    
    def forward(self, spect, audio, speaker_ids=None): # optional spect input
        """
        forward_input[0] = mel_spectrogram:  batch x n_mel_channels x frames
        forward_input[1] = audio: batch x time
        """
        #  Upsample spectrogram to size of audio
        spect = self.upsample(spect) # [B, mels, T//n_group]
        audio, spect = self.squeeze(audio, spect)
        
        output_audio = []
        split_sections = [self.n_early_size, self.n_group]
        for k, (convinv, affine_coup) in enumerate(zip(self.convinv, self.WN)):
            if k % self.n_early_every == 0 and k > 0:
                split_sections[1] -= self.n_early_size
                early_output, audio = audio.split(split_sections, 1)
                # these 2 lines actually copy tensors, may need optimization in the future
                output_audio.append(early_output)
                audio = audio.clone()
            
            audio, log_det_W = convinv(audio)
            
            audio, log_s = affine_coup(audio, spect, speaker_ids=speaker_ids)
            if k:
                logdet += log_det_W + log_s.sum((1, 2))
            else:
                logdet = log_det_W + log_s.sum((1, 2))
        
        assert split_sections[1] == self.z_split_sizes[-1]
        output_audio.append(audio)
        return torch.cat(output_audio, 1).transpose(1, 2).contiguous().view(audio.shape[0], -1), logdet
    
    
    def inverse(self, z, spect, speaker_ids=None):
        batch_dim = spect.shape[0]
        spect = self.upsample(spect)
        z, spect = self.squeeze(z, spect)
        assert z.size(2) <= spect.size(2)
        spect = spect[..., :z.size(2)]
        
        remained_z = []
        for r in z.split(self.z_split_sizes, 1):
            remained_z.append(r.clone())
        *remained_z, z = remained_z
        
        for k, invconv, affine_coup in zip(range(self.n_flows - 1, -1, -1), self.convinv[::-1], self.WN[::-1]):
            
            z, log_s = affine_coup.inverse(z, spect, speaker_ids=speaker_ids)
            z, log_det_W = invconv.inverse(z)
            
            if k == self.n_flows - 1:
                logdet = log_det_W + log_s.sum((1, 2))
            else:
                logdet += log_det_W + log_s.sum((1, 2))
            
            if k % self.n_early_every == 0 and k:
                z = torch.cat((remained_z.pop(), z), 1)
        
        z = z.transpose(1, 2).contiguous().view(batch_dim, -1)
        return z, logdet
    
    @torch.no_grad()
    def infer(self, spect, speaker_ids=None, sigma=1.):
        if len(spect.shape) == 2:
            spect = spect[None, ...]
        
        batch_dim, n_mel_channels, steps = spect.shape
        samples = steps * self.hop_length
        
        z = spect.new_empty((batch_dim, samples)).normal_(std=sigma)
        # z = torch.randn(batch_dim, self.n_group, group_steps, dtype=spect.dtype, device=spect.device).mul_(sigma)
        audio, _ = self.inverse(z, spect, speaker_ids)
        return audio


if __name__ == '__main__':
    import librosa
    import matplotlib.pyplot as plt

    spect, sr = librosa.load(librosa.util.example_audio_file())
    # spect = librosa.feature.melspectrogram(spect=spect, sr=sr, n_fft=1024, hop_length=256, n_mel_channels=80)
    # print(spect.shape, spect.max())
    # plt.imshow(spect ** 0.1, aspect='auto', origin='lower')
    # plt.show()

    spect = torch.Tensor(spect)
    net = WaveGlow(12, 8, 4, 2, sr, 1024, 256, 80, n_layers=5, residual_channels=64, dilation_channels=64,
                   skip_channels=64, bias=True)
    # print(net)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad), "of parameters.")

    spect = net.get_mel(spect[None, ...])[0]
    print(spect.shape, spect.max())
    plt.imshow(spect.numpy(), aspect='auto', origin='lower')
    plt.show()

    audio = torch.rand(2, 16000) * 2 - 1
    z, *_ = net(audio)
    print(z.shape)

    audio = net.infer(spect[:, :10])
    print(audio.shape)
