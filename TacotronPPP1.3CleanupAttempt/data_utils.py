import random
import os
import re
import numpy as np
import torch
import torch.utils.data
import librosa

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence
from yin import compute_yin


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text and speaker ids
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms and f0s from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams, check_files=True, TBPTT=True, speaker_ids=None, verbose=False):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.load_alignments = hparams.load_alignments
        self.truncated_length = hparams.truncated_length
        self.batch_size = hparams.batch_size
        self.speaker_ids = speaker_ids
        if speaker_ids is None:
            self.speaker_ids = self.create_speaker_lookup_table(self.audiopaths_and_text)
        
        self.load_torchmoji = hparams.torchMoji_training and hparams.torchMoji_linear
        
        # ---------- CHECK FILES --------------
        self.start_token = hparams.start_token
        self.stop_token = hparams.stop_token
		if hparams.check_dataset:
	        self.checkdataset(show_info=hparams.checkdataset_show_info, show_warning=hparams.checkdataset_show_warnings)
        # -------------- CHECK FILES --------------
        
		# init STFT
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.f0_min = hparams.f0_min
        self.f0_max = hparams.f0_max
        self.harm_thresh = hparams.harm_thresh
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        
        # Apply weighting to MLP Datasets
        duplicated_audiopaths = [x for x in self.audiopaths_and_text if "SlicedDialogue" in x[0]]
        for i in range(3):
            self.audiopaths_and_text.extend(duplicated_audiopaths)
        
        # SHUFFLE audiopaths
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)
        
        # -------------- PREDICT LENGTH (TBPTT) --------------
        # This bit will simulate the entire epoch and identify what clips to be loaded when, and what time to preserve decoder state.
        # That information is kept in self.dataloader_indexes as an array
        # Interally each element looks like [index_for_audiopaths_and_text, starting_frame_for_spectrogram]
        self.batch_size = hparams.batch_size if speaker_ids is None else hparams.val_batch_size
        n_gpus = hparams.n_gpus
        self.rank = hparams.rank
        self.total_batch_size = self.batch_size * n_gpus # number of audio files being processed together
        self.truncated_length = hparams.truncated_length # frames
        
        self.dataloader_indexes = []
        
        audio_lengths = torch.tensor([self.get_mel(x[0]).shape[1] for x in self.audiopaths_and_text])
        
        batch_remaining_lengths = audio_lengths[:self.total_batch_size]
        batch_frame_offset = torch.zeros(self.total_batch_size)
        batch_indexes = torch.tensor(list(range(self.total_batch_size)))
        processed = 0
        currently_empty_lengths = 0
        
        while audio_lengths.shape[0]+1>processed+self.total_batch_size+currently_empty_lengths:
            # replace empty lengths
            currently_empty_lengths = (batch_remaining_lengths<1).sum().item()
            # update batch_indexes
            batch_indexes[batch_remaining_lengths<1] = torch.arange(processed+self.total_batch_size, processed+self.total_batch_size+currently_empty_lengths)
            # update batch_frame_offset
            batch_frame_offset[batch_remaining_lengths<1] = 0
            # update batch_remaining_lengths
            try:
                batch_remaining_lengths[batch_remaining_lengths<1] = audio_lengths[processed+self.total_batch_size:processed+self.total_batch_size+currently_empty_lengths]
            except RuntimeError: # exception occurs when shape does not match (which happens when we run out of new files to process)
                break
            
            # update how many audiofiles have been fully used
            processed+=currently_empty_lengths
            
            self.dataloader_indexes.extend(list(zip(batch_indexes.numpy(), batch_frame_offset.numpy())))
            #print(batch_remaining_lengths, batch_indexes, sep="\n")
            
            batch_remaining_lengths = batch_remaining_lengths - self.truncated_length # truncate batch
            batch_frame_offset = batch_frame_offset + self.truncated_length
            #print(batch_remaining_lengths, "---------------------", sep="\n")
        
        self.len = len(self.dataloader_indexes)
        # -------------- PREDICT LENGTH (TBPTT) --------------
    
    def checkdataset(self, show_info=False, show_warning=True):
        print("Checking dataset files...", end="")
        audiopaths_length = len(self.audiopaths_and_text)
        filtered_chars=["☺","␤"]
        banned_strings = ["[","]"]
        banned_paths = ["_Mane 6_","_Mane6_"]
        music_stuff = True
        start_token = self.start_token
        stop_token = self.stop_token
        for index, file in enumerate(self.audiopaths_and_text): # index must use seperate iterations from remove
            if music_stuff and r"Songs/" in file[0]:
                self.audiopaths_and_text[index][1] = "♫" + self.audiopaths_and_text[index][1] + "♫"
            self.audiopaths_and_text[index][1] = start_token + self.audiopaths_and_text[index][1] + stop_token
            for filtered_char in filtered_chars:
                self.audiopaths_and_text[index][1] = self.audiopaths_and_text[index][1].replace(filtered_char,"")
        i = 0
        i_offset = 0
        for i_ in range(len(self.audiopaths_and_text)):
            i = i_ + i_offset # iterating on an array you're also updating will cause some indexes to be skipped.
            if i == len(self.audiopaths_and_text): break
            file = self.audiopaths_and_text[i]
            if self.load_mel_from_disk and '.wav' in file[0]:
                if show_warning:
                    print("|".join(file), "\n[warning] in filelist while expecting '.npy' . Being Ignored.")
                self.audiopaths_and_text.remove(file)
                i_offset-=1
                continue
            elif not self.load_mel_from_disk and '.npy' in file[0]:
                if show_warning:
                    print("|".join(file), "\n[warning] in filelist while expecting '.wav' . Being Ignored.")
                self.audiopaths_and_text.remove(file)
                i_offset-=1
                continue
            if not os.path.exists(file[0]):
                if show_warning:
                    print("|".join(file), "\n[warning] does not exist and has been ignored")
                self.audiopaths_and_text.remove(file)
                i_offset-=1
                continue
            if not len(file[1]):
                if show_warning:
                    print("|".join(file), "\n[warning] has no text and has been ignored.")
                self.audiopaths_and_text.remove(file)
                i_offset-=1
                continue
            if len(file[1]) < 3:
                if show_info:
                    print("|".join(file), "\n[info] has no/very little text.")
            if not ((file[1].strip())[-1] in r"!?,.;:♫␤"):
                if show_info:
                    print("|".join(file), "\n[info] has no ending punctuation.")
            if self.load_mel_from_disk:
                melspec = torch.from_numpy(np.load(file[0], allow_pickle=True))
                mel_length = melspec.shape[1]
                if mel_length == 0:
                    print("|".join(file), "\n[warning] has 0 duration and has been ignored")
                    self.audiopaths_and_text.remove(file)
                    i_offset-=1
                    continue
            if any(i in file[1] for i in banned_strings):
                if show_info:
                    print("|".join(file), "\n[info] is in banned strings and has been ignored.")
                self.audiopaths_and_text.remove(file)
                i_offset-=1
                continue
            if any(i in file[0] for i in banned_paths):
                if show_info:
                    print("|".join(file), "\n[info] is in banned paths and has been ignored.")
                self.audiopaths_and_text.remove(file)
                i_offset-=1
                continue
        print("Done")
        print(audiopaths_length, "files in metadata file")
        print(len(self.audiopaths_and_text), "remaining.")
    
    def create_speaker_lookup_table(self, audiopaths_and_text):
        speaker_ids = np.sort(np.unique([x[2] for x in audiopaths_and_text]))
        d = {int(speaker_ids[i]): i for i in range(len(speaker_ids))}
        return d
    
    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            audio_norm = None
            melspec = torch.from_numpy(np.load(filename, allow_pickle=True))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))
        return melspec, audio_norm
    
    def get_f0(self, filepath, audio_norm=None):
        if audio_norm is None:
            audio, sampling_rate = load_wav_to_torch(filepath)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
        
        f0 = self.calc_f0(audio.cpu().numpy(), self.sampling_rate,
                         self.filter_length, self.hop_length, self.f0_min,
                         self.f0_max, self.harm_thresh)
        f0 = torch.from_numpy(f0)[None]
        return f0
    
    def calc_f0(self, audio, sampling_rate=48000, frame_length=2400,
               hop_length=600, f0_min=100, f0_max=300, harm_thresh=0.1):
        f0, harmonic_rates, argmins, times = compute_yin(
            audio, sampling_rate, frame_length, hop_length, f0_min, f0_max,
            harm_thresh)
        pad = int((frame_length / hop_length) / 2)
        f0 = [0.0] * pad + f0 + [0.0] * pad

        f0 = np.array(f0, dtype=np.float32)
        return f0
    
    def get_data(self, index):
        filelist_index, spectrogram_offset = self.dataloader_indexes[index]
        next_filelist_index, next_spectrogram_offset = self.dataloader_indexes[index+self.total_batch_size] if index+self.total_batch_size < self.len else (None, None)
        if filelist_index != next_filelist_index: # if same item in previous minibatch uses same file, preserve decoder state
            preserve_decoder_state = torch.tensor(False)
        else:
            preserve_decoder_state = torch.tensor(True)
    	
		# get line from text file (split by '|')
        audiopath, text, speaker = self.audiopaths_and_text[filelist_index]
		
        text = self.get_text(text) # convert text into tensor representation
        
        mel, audio_norm = self.get_mel(audiopath) # get mel-spec as tensor from audiofile.
        mel = mel[..., int(spectrogram_offset):int(spectrogram_offset+self.truncated_length)] # take the relavent truncated segment
        
        speaker_id = self.get_speaker_id(speaker) # get speaker_id as tensor between  0 -> len(speaker_ids)
        
		torchmoji = self.get_torchmoji_hidden(audiopath) # returns torchMoji hidden if self.load_alignments else None
        
		align_path = f"{audiopath}.align{text.shape[0]}.npy"
        alignment = self.get_alignment(align) # returns alignment if self.load_alignments else None
		
        f0 = get_f0(audiopath, audio_norm):
        f0 = f0[:, int(spectrogram_offset):int(spectrogram_offset)+mel.size(1)]
        
        return (text, mel, speaker_id, torchmoji, preserve_decoder_state, alignment, f0)
    
    
    def get_torchmoji_hidden(self, audiopath):
        if self.load_torchmoji:
	        audiopath_without_ext = ".".join(audiopath.split(".")[:-1])
	        path_path_len = min(len(audiopath_without_ext), 999)
	        file_path_safe = audiopath_without_ext[0:path_path_len]
	        hidden_state = np.load(file_path_safe + "_.npy")
	        return torch.from_numpy(hidden_state).float()
		else:
			return None
    
    
    def get_speaker_id(self, speaker_id):
        return torch.IntTensor([self.speaker_ids[int(speaker_id)]])
	
    
    def get_alignment(self, filename):
        alignment = None
        if self.load_alignments:
	        alignment = torch.from_numpy(np.load(filename))
            assert alignment.shape[-1] == mel.shape[-1], f"Length of alignment ({alignment.shape[-1]}) and mel ({mel.shape[-1]}) do not match for {audiopath}" # assert same length
            assert alignment.shape[0] == text.shape[0], f"length of alignment encode dim ({alignment.shape[0]}) and text ({text.shape[0]}) do not match for {audiopath}" # assert same num of chars
        return alignment
    
    
    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm
    
    
    def __getitem__(self, index):
        return self.get_data(index)
    
    
    def __len__(self):
        return self.len


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step, load_alignments):
        self.n_frames_per_step = n_frames_per_step
        self.load_alignments = load_alignments

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [texts, mels, speaker_ids, torchmoji_hidden, preserve_decoder, pag_alignments, f0s]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]
        
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text
        
        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0
        
        # include mel padded, gate padded and speaker ids
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        speaker_ids = torch.LongTensor(len(batch))
		
		# (optional) TorchMoji hidden state
        if batch[0][3] is not None:# if torchmoji hidden in first item in batch is not None
            torchmoji_hidden = torch.FloatTensor(len(batch), batch[0][3].shape[0])
        else:
            torchmoji_hidden = None
		
		# Truncated minibatches with resets
        preserve_decoder_states = torch.FloatTensor(len(batch))
		
		# (optional) Pre-Alignment Guided Attention
        if self.load_alignments:
            align_padded = torch.FloatTensor(len(batch), max_input_len, max_target_len)
            align_padded.zero_()
            max_align_len = max([x[2].size(1) for x in batch])
            assert max_align_len == max_target_len # ensure loaded attention target matches mel len
            max_align_enc = max([x[2].size(0) for x in batch])
            assert max_align_enc == max_input_len # ensure loaded attention target matches input text len
        else:
            align_padded = None
		
        # f0s
        f0_padded = torch.FloatTensor(len(batch), 1, max_target_len)
        f0_padded.zero_()
        
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
			# mel
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
			
			# gate
            gate_padded[i, mel.size(1)-1:] = 1
             # if preserving decoder, then clip is not ending and gate should be all 0's
			gate_padded[i, -1] = not batch[ids_sorted_decreasing[i]][4]
            
			# output_lengths
            output_lengths[i] = mel.size(1)
			
			# speaker_ids
            speaker_ids[i] = batch[ids_sorted_decreasing[i]][2]
			
			# torchmoji_hidden
            if torchmoji_hidden is not None:
                torchmoji_hidden[i] = batch[ids_sorted_decreasing[i]][3]
			
			# preserve decoder
            preserve_decoder_states[i] = batch[ids_sorted_decreasing[i]][4]
			
			# pag alignments
            if self.load_alignments:
                alignment = batch[ids_sorted_decreasing[i]][5]
                align_padded[i, :alignment.size(0), :mel.size(1)] = alignment
            
            # f0s
            f0 = batch[ids_sorted_decreasing[i]][6]
            f0_padded[i, :, :f0.size(1)] = f0
        
        model_inputs = (text_padded, input_lengths, mel_padded, gate_padded, output_lengths,
                        speaker_ids, torchmoji_hidden, preserve_decoder_states, align_padded, f0_padded)
        return model_inputs
