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
import os
import random
import argparse
import json
import torch
import torch.utils.data
import sys
import numpy as np
from scipy.io.wavfile import read
from math import ceil
from pathlib import Path

# We're using the audio processing from TacoTron2 to make sure it matches
from utils.layers import TacotronSTFT
from utils.empthasis import PreEmphasis, InversePreEmphasis


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f if len(line.strip())]
    return filepaths_and_text


def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate

#def load_wav_to_torch(full_path):
#    """
#    Loads wavdata into torch array
#    """
#    data, sampling_rate = soundfile.read(full_path)
#    return torch.tensor(data).float(), sampling_rate

def get_mel_from_file(mel_path):
    melspec = np.load(mel_path)
    melspec = torch.autograd.Variable(torch.from_numpy(melspec), requires_grad=False)
    melspec = torch.squeeze(melspec, 0)
    return melspec


def check_files(audio_files, hparams, verbose=False):
    segment_length = hparams.segment_length
    if verbose:
        print("Files before checking: ", len(audio_files))

    i = 0
    i_offset = 0
    for i_ in range(len(audio_files)):
        i = i_ + i_offset
        if i == len(audio_files): break
        file = audio_files[i]
        if not os.path.exists(file[0]):
            print(file[0],"does not exist")
            audio_files.remove(file); i_offset-=1; continue
        try:
            audio_data, sample_r = load_wav_to_torch(file[0])
        except Exception as ex:
            print(ex)
            print(file[0],"failed to load audio.")
            audio_files.remove(file); i_offset-=1; continue
            
        if audio_data.size(0) <= segment_length:
            print(file[0],"is too short")
            audio_files.remove(file); i_offset-=1; continue

    if verbose:
        print("Files after checking: ", len(audio_files))
    return audio_files


class Mel2Samp(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, hparams):
        self.training_files = hparams.training_files
        self.max_wav_value = hparams.max_wav_value
        self.segment_length = hparams.segment_length
        self.filter_length = hparams.filter_length
        self.mel_fmin = hparams.mel_fmin
        self.mel_fmax = hparams.mel_fmax
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.preempthasis = hparams.preempthasis
        self.segment_length = hparams.segment_length
        self.sampling_rate = hparams.sampling_rate
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.audio_files = load_filepaths_and_text(self.training_files)
        self.cache_spectrograms = hparams.cache_spectrograms
        self.n_mel_channels = hparams.n_mel_channels
        
        self.audio_files = check_files(self.audio_files, hparams, verbose=True)
        
        if hparams.speaker_embed_dim: # if multispeaker model
            self.speaker_ids = self.create_speaker_lookup_table(self.audio_files)
        
        random.seed(hparams.seed)
        random.shuffle(self.audio_files)
        
        self.stft = TacotronSTFT(filter_length=self.filter_length,
                                 hop_length=self.hop_length,
                                 win_length=self.win_length,
                                 sampling_rate=self.sampling_rate,
                                 n_mel_channels=hparams.n_mel_channels,
                                 mel_fmin=self.mel_fmin,
                                 mel_fmax=self.mel_fmax)
        
        if self.preempthasis > 0.:
            self.preempthasise = PreEmphasis(self.preempthasis)

    def create_speaker_lookup_table(self, audiopaths_and_text):
        speaker_ids = np.sort(np.unique([x[2] for x in audiopaths_and_text]))
        d = {int(speaker_ids[i]): i for i in range(len(speaker_ids))}
        return d

    def get_speaker_id(self, speaker_id):
        return torch.IntTensor([self.speaker_ids[int(speaker_id)]])

    def get_mel(self, audio):
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm).squeeze(0)
        return melspec
    
    def get_segment(self, audio, mel, segment_length, hop_length, n_mel_channels=160):
        mel_segment_length = int(segment_length/hop_length)
        if audio.size(0) >= segment_length:
            max_mel_start = int((audio.size(0)-segment_length)/hop_length) # audio.size(0)%self.hop_length is the remainder
            mel_start = random.randint(0, max_mel_start)
            audio_start = mel_start*hop_length
            audio = audio[audio_start:audio_start + segment_length]
            mel = mel[:,mel_start:mel_start + mel_segment_length]
        else:
            mel_start = 0
            n_mel_channels = 160 # TODO take from config file
            len_pad = int((segment_length/ hop_length) - mel.shape[1])
            pad = np.ones((n_mel_channels, len_pad), dtype=np.float32) * -11.512925
            mel =  np.append(mel, pad, axis=1)
            audio = torch.nn.functional.pad(audio, (0, segment_length - audio.size(0)), 'constant').data
        return audio, mel, mel_start, mel_start + mel_segment_length
    
    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio, sampling_rate = load_wav_to_torch(filename[0])
        assert audio.shape[0], f"Audio has 0 length.\nFile: {filename[0]}\nIndex: {index}"
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        
        mel = None
        if self.load_mel_from_disk and filename[1] and os.path.exists(filename[1]):
            mel = np.load(filename[1])
        elif self.cache_spectrograms:
            cached_fpath = f"{os.path.splitext(filepath[0])[0]}_{self.n_mel_channels}_{self.hop_length}_{self.win_length}_{len(audio)}_cacheSpec.npy"
            if os.path.exists(cached_fpath):
                mel = np.load(cached_fpath, allow_pickle=True)
            else:
                mel = self.get_mel(audio)
                np.save(cached_fpath, mel.numpy())
        
        if mel is not None: # If loading an already made mel, the audio segment need to split along hop_length intervals to align correctly.
            assert self.segment_length % self.hop_length == 0, 'self.segment_length must be n times of self.hop_length'
            # Take segment
            loop = 0
            while True:
                audio_segment, mel_segment, start_step, stop_step = self.get_segment(audio, mel, self.segment_length, self.hop_length)
                std = torch.std(audio_segment)
                if std > 250: break # if sample is not silent, continue.
                loop+=1
                if loop > 20:
                    print("No Silent Sample Found, filename:",filename[0]); break
            audio, mel = audio_segment, mel_segment
            mel = torch.from_numpy(mel).float()
        else:
            # Take segment
            if audio.size(0) >= self.segment_length:
                max_audio_start = audio.size(0) - self.segment_length
                std = 9e9
                loop = 0
                while True:
                    audio_start = random.randint(0, max_audio_start)
                    audio_segment = audio[audio_start:audio_start + self.segment_length]
                    std = torch.std(audio_segment)
                    if std > 250: break # if sample is not silent, use sample for WaveGlow.
                    loop+=1
                    if loop > 20:
                        print("No Silent Sample Found, filename:",filename[0]); break
                audio = audio_segment
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data
            assert audio.shape[0], f"Audio has 0 length.\nFile: {filename[0]}\nIndex: {index}"
            mel = self.get_mel(audio) # generate mel from audio segment
        
        audio = audio / self.max_wav_value
        
        if hasattr(self, 'preempthasise'):
            audio = self.preempthasise(audio.unsqueeze(0).unsqueeze(0)).squeeze()
        
        speaker_id = self.get_speaker_id(filename[2])
        
        return (mel, audio, speaker_id) # (mel, audio, speaker_id)

    def __len__(self):
        return len(self.audio_files)