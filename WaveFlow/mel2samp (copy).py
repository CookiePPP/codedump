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

# We're using the audio processing from TacoTron2 to make sure it matches
sys.path.insert(0, 'tacotron2')
from tacotron2.layers import TacotronSTFT

# utils
from waveglow_utils import PreEmphasis, InversePreEmphasis

MAX_WAV_VALUE = 32768.0

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate


def get_mel_from_file(mel_path):
    melspec = np.load(mel_path)
    melspec = torch.autograd.Variable(torch.from_numpy(melspec), requires_grad=False)
    melspec = torch.squeeze(melspec, 0)
    return melspec

class Mel2Samp(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, training_files, validation_files, validation_windows, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax, load_mel_from_disk, preempthasis):
        self.audio_files = load_filepaths_and_text(training_files)
        
        print("Files before checking: ", len(self.audio_files))
        
        i = 0
        i_offset = 0
        for i_ in range(len(self.audio_files)):
            i = i_ + i_offset
            if i == len(self.audio_files): break
            file = self.audio_files[i]
            if not os.path.exists(file[0]):
                print(file[0],"does not exist")
                self.audio_files.remove(file); i_offset-=1; continue
            
            audio_data, sample_r = load_wav_to_torch(file[0])
            if audio_data.size(0) <= segment_length:
                print(file[0],"is too short")
                self.audio_files.remove(file); i_offset-=1; continue
        
        print("Files after checking: ", len(self.audio_files))
        
        self.load_mel_from_disk = load_mel_from_disk
        self.speaker_ids = self.create_speaker_lookup_table(self.audio_files)
        random.seed(1234)
        random.shuffle(self.audio_files)
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 n_mel_channels=160,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        if preempthasis:
            self.preempthasise = PreEmphasis(preempthasis)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.win_length = win_length
    
    def create_speaker_lookup_table(self, audiopaths_and_text):
        speaker_ids = np.sort(np.unique([x[2] for x in audiopaths_and_text]))
        d = {int(speaker_ids[i]): i for i in range(len(speaker_ids))}
        return d
    
    def get_speaker_id(self, speaker_id):
        return torch.IntTensor([self.speaker_ids[int(speaker_id)]])
    
    def get_mel(self, audio):
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm).squeeze(0)
        return melspec
    
    def get_segment(self, audio, mel, segment_length, hop_length, n_mel_channels=160):
        mel_segment_length = int(segment_length/hop_length) # 8400/600 = 14
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
        
        if (self.load_mel_from_disk):
            # Take segment
            mel = np.load(filename[1])
            assert self.segment_length % self.hop_length == 0, 'self.segment_length must be n times of self.hop_length'
#            if (mel.shape[1] > ceil(len(audio)/self.hop_length)):
#                print('mel is longer than audio file')
#                print('path', filename[1], '\nmel_length', mel.shape[1], '\naudio length', len(audio), '\naudio_hops', ceil(len(audio)/self.hop_length))
#                raise Exception
#            if (mel.shape[1] < ceil(len(audio)/self.hop_length)):
#                print('mel is shorter than audio file')
#                print('path', filename[1], '\nmel_length', mel.shape[1], '\naudio length', len(audio), '\naudio_hops', ceil(len(audio)/self.hop_length))
#                raise Exception
            loop = 0
            while True:
                audio_, mel_, start_step, stop_step = self.get_segment(audio, mel, self.segment_length, self.hop_length) # get random segment of audio file
                std = torch.std(audio_)
                if std > 250: break # if sample is not silent, use sample for WaveGlow.
                loop+=1
                if loop > 20:
                    print("No Silent Sample Found, filename:",filename[0]); break
            #print(f"STD: {std} Loops: {loop}")
            audio, mel = audio_, mel_
            
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
                audio = audio_
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data
            assert audio.shape[0], f"Audio has 0 length.\nFile: {filename[0]}\nIndex: {index}"
            mel = self.get_mel(audio) # generate mel from audio segment
        
        audio = audio / MAX_WAV_VALUE
        
        if hasattr(self, 'preempthasise'):
            audio = self.preempthasise(audio.unsqueeze(0).unsqueeze(0)).squeeze()
        
        speaker_id = self.get_speaker_id(filename[2])
        
        #mel = (mel+5.2)*0.5 # shift values between approx -4 and 4
        return (mel, audio, speaker_id) # (mel, audio, speaker_id)

    def __len__(self):
        return len(self.audio_files)

# ===================================================================
# Takes directory of clean audio and makes directory of spectrograms
# Useful for making test sets
# ===================================================================
if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Output directory')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    data_config = json.loads(data)["data_config"]
    mel2samp = Mel2Samp(**data_config)

    filepaths = files_to_list(args.filelist_path)

    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)

    for filepath in filepaths:
        audio, sr = load_wav_to_torch(filepath)
        melspectrogram = mel2samp.get_mel(audio)
        filename = os.path.basename(filepath)
        new_filepath = args.output_dir + '/' + filename + '.pt'
        print(new_filepath)
        torch.save(melspectrogram, new_filepath)
