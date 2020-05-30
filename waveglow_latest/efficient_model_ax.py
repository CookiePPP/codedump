import torch
import torch.nn as nn
import torch.nn.functional as F
from efficient_util import add_weight_norms
import numpy as np

from efficient_modules import AffineCouplingBlock, InvertibleConv1x1

class WaveGlow(nn.Module):
    def __init__(self, yoyo, yoyo_WN, n_mel_channels, n_flows, n_group, n_early_every,
                n_early_size, memory_efficient, spect_scaling, upsample_mode, upsample_first, speaker_embed, cond_layers, cond_hidden_channels, cond_output_channels, cond_kernel_size, cond_residual, cond_padding_mode, WN_config, win_length, hop_length):
        super(WaveGlow, self).__init__()
        assert(n_group % 2 == 0)
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.win_size = win_length
        self.hop_length = hop_length
        self.n_mel_channels = n_mel_channels
        self.upsample_first = not upsample_first
        self.upsample_mode = WN_config['upsample_mode']
        
        self.speaker_embed_dim = speaker_embed
        self.multispeaker = self.speaker_embed_dim > 0 or WN_config['speaker_embed_dim'] > 0
        
        if self.speaker_embed_dim:
            max_speakers = 512
            self.speaker_embed = nn.Embedding(max_speakers, self.speaker_embed_dim)
        
        self.cond_residual = cond_residual
        if self.cond_residual: # override conditional output size if using residuals
            cond_output_channels = self.n_mel_channels+self.speaker_embed_dim
        
        self.cond_layers = nn.ModuleList()
        if cond_layers:
            # messy initialization for arbitrary number of layers, input dims and output dims
            cond_kernel_size = 2*cond_kernel_size - 1 # 1 -> 1, 2 -> 3, 3 -> 5
            cond_pad = int((cond_kernel_size - 1)/2)
            dimensions = [self.n_mel_channels+self.speaker_embed_dim,]+[cond_hidden_channels]*(cond_layers-1)+[cond_output_channels,]
            in_dims = dimensions[:-1]
            out_dims = dimensions[1:]
            #print(in_dims, out_dims, "\n")
            for i in range(len(in_dims)):
                indim = in_dims[i]
                outim = out_dims[i]
                cond_layer = nn.Conv1d(indim, outim, cond_kernel_size, padding=cond_pad, padding_mode=cond_padding_mode)# (in_channels, out_channels, kernel_size)
                cond_layer = nn.utils.weight_norm(cond_layer, name='weight')
                self.cond_layers.append(cond_layer)
            WN_cond_channels = cond_output_channels
        else:
            WN_cond_channels = self.n_mel_channels+self.speaker_embed_dim
        
        if yoyo_WN:
            raise NotImplementedError
            from efficient_modules import WN
        else:
            from glow_ax import WN
        
        self.upsample_factor = hop_length // n_group
        sub_win_size = win_length // n_group
        
        self.convinv = nn.ModuleList()
        self.WN = nn.ModuleList()
        
        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = n_group
        self.z_split_sizes = []
        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_remaining_channels -= n_early_size
                self.z_split_sizes.append(n_early_size)
            
            assert n_remaining_channels > 0, "n_remaining_channels is 0. (increase n_group or decrease n_early_every/n_early_size)"
            
            self.convinv.append( InvertibleConv1x1(n_remaining_channels, memory_efficient=memory_efficient) )
            if yoyo_WN:
                self.WN.append( AffineCouplingBlock(WN, memory_efficient=memory_efficient, in_channels=n_remaining_channels//2,
                                    cond_in_channels=WN_cond_channels, **WN_config) )
            else: # I promise these two used to be different
                self.WN.append( AffineCouplingBlock(WN, memory_efficient=memory_efficient, n_in_channels=n_remaining_channels//2,
                                    cond_in_channels=WN_cond_channels, **WN_config) )
        self.z_split_sizes.append(n_remaining_channels)
    
    
    def forward(self, cond, audio, speaker_ids=None): # optional cond input
        """
        forward_input[0] = mel_spectrogram:  batch x n_mel_channels x frames
        forward_input[1] = audio: batch x time
        """
        # Add speaker conditioning
        if self.speaker_embed_dim:
            speaker_embeddings = self.speaker_embed(speaker_ids)
            speaker_embeddings = speaker_embeddings.unsqueeze(-1).repeat(1, 1, cond.shape[2]) # shape like cond
            cond = torch.cat([cond, speaker_embeddings], dim=1) # and concat them
        
        cond_res = cond
        for layer in self.cond_layers:
            cond_res = layer(cond_res)
        
        if self.cond_residual:
            cond = cond + cond_res # adjust the original input by a residual
        else:
            cond = cond_res # completely reform the input into something else
        
        batch_dim, n_mel_channels, group_steps = cond.shape
        audio = audio.view(batch_dim, -1, self.n_group).transpose(1, 2)
        
        #  Upsample spectrogram to size of audio
        if self.upsample_first:
            cond = self._upsample_mels(cond, audio.size(2)) # [B, mels, T//n_group]
        
        #assert audio.size(2) <= cond.size(2)
        #cond = cond[..., :audio.size(2)]
        
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
            
            audio, log_s = affine_coup(audio, cond, speaker_ids=speaker_ids)
            if k:
                logdet += log_det_W + log_s.sum((1, 2))
            else:
                logdet = log_det_W + log_s.sum((1, 2))
        
        assert split_sections[1] == self.z_split_sizes[-1]
        output_audio.append(audio)
        return torch.cat(output_audio, 1).transpose(1, 2).contiguous().view(batch_dim, -1), logdet
    
    def _upsample_mels(self, cond, audio_size):
        cond = F.interpolate(cond, size=audio_size, mode=self.upsample_mode, align_corners=True if self.upsample_mode == 'linear' else None)
        return cond
        # return self.upsampler(cond)
    
    def inverse(self, z, cond, speaker_ids=None):
        # Add speaker conditioning
        if self.speaker_embed_dim:
            speaker_embeddings = self.speaker_embed(speaker_ids)
            speaker_embeddings = speaker_embeddings.unsqueeze(-1).repeat(1, 1, cond.shape[2]) # shape like cond
            cond = torch.cat([cond, speaker_embeddings], dim=1) # and concat them
        
        cond_res = cond
        for layer in self.cond_layers:
            cond_res = layer(cond_res)
        
        if self.cond_residual:
            cond += cond_res # adjust the original input by a residual
        else:
            cond = cond_res # completely reform the input into something else
        
        batch_dim, n_mel_channels, group_steps = cond.shape
        z = z.view(batch_dim, -1, self.n_group).transpose(1, 2)
        
        #  Upsample spectrogram to size of audio
        if self.upsample_first:
            cond = self._upsample_mels(cond, z.size(2)) # [B, mels, T//n_group]
        
        #assert z.size(2) <= cond.size(2)
        #cond = cond[..., :z.size(2)]
        
        remained_z = []
        for r in z.split(self.z_split_sizes, 1):
            remained_z.append(r.clone())
        *remained_z, z = remained_z
        
        logdet = None
        for k, invconv, affine_coup in zip(range(self.n_flows - 1, -1, -1), self.convinv[::-1], self.WN[::-1]):
            
            z, log_s = affine_coup.inverse(z, cond, speaker_ids=speaker_ids)
            z, log_det_W = invconv.inverse(z)
            
            #if k == self.n_flows - 1:
            #    logdet = log_det_W + log_s.sum((1, 2))
            #else:
            #    logdet += log_det_W + log_s.sum((1, 2))
            
            if k % self.n_early_every == 0 and k:
                z = torch.cat((remained_z.pop(), z), 1)
        
        z = z.transpose(1, 2).contiguous().view(batch_dim, -1)
        return z, logdet
    
    @torch.no_grad()
    def infer(self, spect, speaker_ids=None, artifact_trimming=1, sigma=1.):
        if len(spect.shape) == 2:
            spect = spect[None, ...] # [n_mel, T//hop_length] -> [B, n_mel, T//hop_length]
        if artifact_trimming:
            spect = F.pad(spect, (0, artifact_trimming), value=-11.512925)
        
        batch_dim, n_mel_channels, steps = spect.shape # [B, n_mel, T//hop_length]
        samples = steps * self.hop_length # T = T//hop_length * hop_length
        
        z = spect.new_empty((batch_dim, samples)) # [B, T]
        if sigma > 0:
            z.normal_(std=sigma)
        audio, _ = self.inverse(z, spect, speaker_ids)
        if artifact_trimming:
            audio_trim = artifact_trimming*self.hop_length # amount of audio to trim
            audio = audio[:, :-audio_trim]
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
