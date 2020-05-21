import os
import random
import argparse
import json
import torch
import torch.utils.data
import traceback
import sys
import numpy as np
from scipy.io.wavfile import read
from math import ceil
from pathlib import Path

class hparams_class:
    def __init__(self):
        self.training_files = "code_tests/test_materials/filelists/validation_utf8.txt"
        self.max_wav_value = float(2**15)
        self.sampling_rate = 48000
        self.filter_length = 2400
        self.hop_length = 600
        self.win_length = 2400
        self.mel_fmin = 0
        self.mel_fmax = 16000
        self.load_mel_from_disk = False
        self.segment_length = 24000
        self.preempthasis = 0.00
        self.cache_spectrograms = False
        self.n_mel_channels = 160
        self.speaker_embed_dim = 1
        self.seed = 1234

def test_mel2samp():
    """Test mel2samp modules on example data."""
    from mel2samp import Mel2Samp
    
    hparams = hparams_class()
    
    passed = 0
    
    
    # test filelist loader
    try:
        from mel2samp import load_filepaths_and_text
        audio_files = load_filepaths_and_text("code_tests/test_materials/filelists/validation_utf8.txt")
        passed+=1
        print("--PASSED--\n")
    except Exception as ex:
        print("--EXCEPTION-- @ Load Filepaths and Text (UTF-8)")
        traceback.print_exc(file=sys.stdout)
        print("\n")
    
    
    # test filelist checker
    try:
        assert audio_files
        from mel2samp import check_files
        audio_files = check_files(audio_files, hparams)
        assert len(audio_files) == 1
        passed+=1
        print("--PASSED--\n")
        del audio_files
    except Exception as ex:
        print("--EXCEPTION-- @ Load Filepaths and Text (UTF-8)")
        traceback.print_exc(file=sys.stdout)
        print("\n")
    
    
    # test initalization
    try:
        trainset = Mel2Samp(hparams)
        passed+=1
        print("--PASSED--\n")
    except Exception as ex:
        print("--EXCEPTION-- @ Mel2Samp Initialization")
        traceback.print_exc(file=sys.stdout)
        print("\n")
    
    
    # test 16-BIT .wav to torch
    try:
        from mel2samp import load_wav_to_torch
        x, sr = load_wav_to_torch("code_tests/test_materials/audio_0/example_16bits.wav")
        assert len(x)
        assert x.max() <= 2**15
        assert x.min() >= -(2**15)
        assert sr == 48000
        passed+=1
        print("--PASSED--\n")
    except Exception as ex:
        print("--EXCEPTION-- @ Load 16-BIT .wav to Pytorch")
        traceback.print_exc(file=sys.stdout)
        print("\n")
    
    
    # test 24-BIT .wav to torch
    try:
        from mel2samp import load_wav_to_torch
        x, sr = load_wav_to_torch("code_tests/test_materials/audio_0/example_24bits.wav")
        assert len(x)
        assert x.max() <= 2**23
        assert x.min() >= -(2**23)
        assert sr == 48000
        passed+=1
        print("--PASSED--\n")
    except Exception as ex:
        print("--EXCEPTION-- @ Load 24-BIT .wav to Pytorch")
        traceback.print_exc(file=sys.stdout)
        print("\n")
    
    
    # test 32-BIT .wav to torch
    try:
        from mel2samp import load_wav_to_torch
        x, sr = load_wav_to_torch("code_tests/test_materials/audio_0/example_32bits.wav")
        assert len(x)
        assert x.max() <= 2**31
        assert x.min() >= -(2**31)
        assert sr == 48000
        passed+=1
        print("--PASSED--\n")
    except Exception as ex:
        print("--EXCEPTION-- @ Load 32-BIT .wav to Pytorch")
        traceback.print_exc(file=sys.stdout)
        print("\n")
    
    
    # test 32-BIT .mp3 to torch
    try:
        from mel2samp import load_wav_to_torch
        x, sr = load_wav_to_torch("code_tests/test_materials/audio_0/example_32bits.mp3")
        assert len(x)
        assert x.max() <= 2**31
        assert x.min() >= -(2**31)
        assert sr == 48000
        passed+=1
        print("--PASSED--\n")
    except Exception as ex:
        print("--EXCEPTION-- @ Load 32-BIT .mp3 to Pytorch")
        traceback.print_exc(file=sys.stdout)
        print("\n")
    
    
    # test 16-BIT .wav to mel
    try:
        x, sr = load_wav_to_torch("code_tests/test_materials/audio_0/example_16bits.wav")
        x = trainset.get_mel(x)
        passed+=1
        print("--PASSED--\n")
    except Exception as ex:
        print("--EXCEPTION-- @ 16-BIT .wav to Mel-spec")
        traceback.print_exc(file=sys.stdout)
        print("\n")
    
    
    # test 24-BIT .wav to mel
    try:
        x, sr = load_wav_to_torch("code_tests/test_materials/audio_0/example_24bits.wav")
        x = trainset.get_mel(x)
        passed+=1
        print("--PASSED--\n")
    except Exception as ex:
        print("--EXCEPTION-- @ 24-BIT .wav to Mel-spec")
        traceback.print_exc(file=sys.stdout)
        print("\n")
    
    
    # test 32-BIT .wav to mel
    try:
        x, sr = load_wav_to_torch("code_tests/test_materials/audio_0/example_32bits.wav")
        x = trainset.get_mel(x)
        passed+=1
        print("--PASSED--\n")
    except Exception as ex:
        print("--EXCEPTION-- @ 32-BIT .wav to Mel-spec")
        traceback.print_exc(file=sys.stdout)
        print("\n")
    
    
    # test 32-BIT .mp3 to mel
    try:
        x, sr = load_wav_to_torch("code_tests/test_materials/audio_0/example_32bits.mp3")
        x = trainset.get_mel(x)
        passed+=1
        print("--PASSED--\n")
    except Exception as ex:
        print("--EXCEPTION-- @ 32-BIT .mp3 to Mel-spec")
        traceback.print_exc(file=sys.stdout)
        print("\n")
    
    
    # test __getitem__ with load_mel_from_disk = False
    try:
        assert trainset # This test will fail if Mel2Samp cannot initalize
        trainset.load_mel_from_disk = False
        trainset.__getitem__(0)
        passed+=1
        print("--PASSED--\n")
    except Exception as ex:
        print("--EXCEPTION-- @  __getitem__ with load_mel_from_disk = False")
        traceback.print_exc(file=sys.stdout)
        print("\n")
    
    
    # test __getitem__ with load_mel_from_disk = True
    try:
        assert trainset # This test will fail if Mel2Samp cannot initalize
        trainset.load_mel_from_disk = True
        trainset.__getitem__(0)
        passed+=1
        print("--PASSED--\n")
    except Exception as ex:
        print("--EXCEPTION-- @  __getitem__ with load_mel_from_disk = True")
        traceback.print_exc(file=sys.stdout)
        print("\n")
    
    
    # test initalization with Pre-empthasis
    try:
        trainset = None
        hparams.preempthasis = 0.98
        trainset = Mel2Samp(hparams)
        passed+=1
        print("--PASSED--\n")
    except Exception as ex:
        print("--EXCEPTION-- @ Mel2Samp with Pre-empthasis Initialization")
        traceback.print_exc(file=sys.stdout)
        print("\n")
    
    
    # test __getitem__ with Pre-empthasis
    try:
        assert trainset # This test will fail if Mel2Samp cannot initalize
        trainset.load_mel_from_disk = False
        trainset.__getitem__(0)
        passed+=1
        print("--PASSED--\n")
    except Exception as ex:
        print("--EXCEPTION-- @  __getitem__ with Pre-empthasis")
        traceback.print_exc(file=sys.stdout)
        print("\n")