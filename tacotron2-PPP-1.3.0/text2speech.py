import os
import numpy as np
import sys
import time
import argparse
import torch
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from model import Tacotron2
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser
from utils import load_filepaths_and_text
import json
import difflib

class T2S:
    def __init__(self):
        # load T2S config
        with open('t2s_config.json', 'r') as f:
            self.config = json.load(f)
        
        speaker_ids_fpath = self.config.get('model').get('speaker_ids_file')
        waveglow_path = self.config.get('model').get('waveglow')
        waveglow_confpath = self.config.get('model').get('waveglowconfig')
        tacotron_path = self.config.get('model').get('tacotron2')
        
        self.tacotron, self.tt_hparams, self.tt_sp_name_lookup, self.tt_sp_id_lookup = self.load_tacotron2(tacotron_path)
        
        self.waveglow, self.wg_train_sigma, self.wg_sp_id_lookup = self.load_waveglow(waveglow_path, waveglow_confpath)
        
        if self.tt_hparams.torchMoji_linear: # if Tacotron includes a torchMoji layer
            self.tm_sentence_tokenizer, self.tm_torchmoji = self.load_torchmoji()
        
        # override since my checkpoints are damaged
        self.tt_sp_name_lookup = {name: self.tt_sp_id_lookup[int(ext_id)] for _, name, ext_id in load_filepaths_and_text(speaker_ids_fpath)}
        
        print("T2S Initialized and Ready!")
    
    def load_torchmoji(self):
        """ Use torchMoji to score texts for emoji distribution.
        
        The resulting emoji ids (0-63) correspond to the mapping
        in emoji_overview.png file at the root of the torchMoji repo.
        
        Writes the result to a csv file.
        """
        import json
        import numpy as np
        import os
        from torchmoji.sentence_tokenizer import SentenceTokenizer
        from torchmoji.model_def import torchmoji_feature_encoding
        from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
        
        print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
        with open(VOCAB_PATH, 'r') as f:
            vocabulary = json.load(f)
        
        maxlen = 130
        texts = ["Testing!",]
        
        with torch.no_grad():
            # init model
            st = SentenceTokenizer(vocabulary, maxlen, ignore_sentences_with_only_custom=True)
            torchmoji = torchmoji_feature_encoding(PRETRAINED_PATH)
        return st, torchmoji
    
    
    def get_torchmoji_hidden(self, texts):
        with torch.no_grad():
            tokenized, _, _ = self.tm_sentence_tokenizer.tokenize_sentences(texts) # input array [B] e.g: ["Test?","2nd Sentence!"]
            embedding = self.tm_torchmoji(tokenized) # returns np array [B, Embed]
        return embedding
    
    
    def load_waveglow(self, waveglow_path, config_fpath):
        # Load config file
        with open(config_fpath) as f:
            data = f.read()
        config = json.loads(data)
        train_config = config["train_config"]
        global data_config
        data_config = config["data_config"]
        global dist_config
        dist_config = config["dist_config"]
        global waveglow_config
        waveglow_config = {
            **config["waveglow_config"], 
            'win_length': data_config['win_length'],
            'hop_length': data_config['hop_length']
        }
        print(waveglow_config)
        print(f"Config File from '{config_fpath}' successfully loaded.")
        
        # import the correct model core
        if waveglow_config["yoyo"]:
            from efficient_model import WaveGlow
        else:
            from glow import WaveGlow
        
        # initialize model
        print(f"intializing WaveGlow model... ", end="")
        waveglow = WaveGlow(**waveglow_config).cuda()
        print(f"Done!")
        
        # load checkpoint from file
        print(f"loading WaveGlow checkpoint... ", end="")
        checkpoint = torch.load(waveglow_path)
        waveglow.load_state_dict(checkpoint['model']) # and overwrite initialized weights with checkpointed weights
        waveglow.cuda().eval().half() # move to GPU and convert to half precision
        print(f"Done!")
        
        print(f"initializing Denoiser... ", end="")
        denoiser = Denoiser(waveglow)
        print(f"Done!")
        waveglow_iters = checkpoint['iteration']
        print(f"WaveGlow trained for {waveglow_iters} iterations")
        speaker_lookup = checkpoint['speaker_lookup'] # ids lookup
        training_sigma = train_config['sigma']
        
        return waveglow, training_sigma, speaker_lookup
    
    
    def load_tacotron2(self, tacotron_path):
        """Loads tacotron2,
        Returns:
        - model
        - hparams
        - speaker_lookup
        """
        checkpoint = torch.load(tacotron_path) # load file into memory
        print("Loading Tacotron... ", end="")
        checkpoint_hparams = checkpoint['hparams'] # get hparams
        checkpoint_dict = checkpoint['state_dict'] # get state_dict
        
        model = load_model(checkpoint_hparams) # initialize the model
        model.load_state_dict(checkpoint_dict) # load pretrained weights
        _ = model.cuda().eval().half()
        print("Done")
        tacotron_speaker_name_lookup = checkpoint['speaker_name_lookup'] # save speaker name lookup
        tacotron_speaker_id_lookup = checkpoint['speaker_id_lookup'] # save speaker_id lookup
        print(f"This Tacotron model has been trained for {checkpoint['iteration']} Iterations.")
        return model, checkpoint_hparams, tacotron_speaker_name_lookup, tacotron_speaker_id_lookup
    
    
    def get_wg_sp_id_from_tt_sp_names(self, names):
        """Get WaveGlow speaker ids from Tacotron2 named speaker lookup. (This should function should be removed once WaveGlow has named speaker support)."""
        tt_model_ids = [self.tt_sp_name_lookup[name] for name in names]
        reversed_lookup = {v: k for k, v in self.tt_sp_id_lookup.items()}
        tt_ext_ids = [reversed_lookup[int(speaker_id)] for speaker_id in tt_model_ids]
        wv_model_ids = [self.wg_sp_id_lookup[int(speaker_id)] for speaker_id in tt_ext_ids]
        return wv_model_ids
    
    
    def get_closest_names(self, names):
        possible_names = list(self.tt_sp_name_lookup.keys())
        validated_names = [difflib.get_close_matches(name, possible_names, n=2, cutoff=0.01)[0] for name in names] # change all names in input to the closest valid name
        return validated_names
    
    
    def infer(self, text, speaker_names, style_mode, gate_delay=6, max_decoder_steps=1600, gate_threshold=0.6, filename=None):
        with torch.no_grad():
            self.tacotron.decoder.gate_delay = gate_delay
            self.tacotron.decoder.max_decoder_steps = max_decoder_steps
            self.tacotron.decoder.gate_threshold = gate_threshold
            
            # find closest valid name
            speaker_names = self.get_closest_names(speaker_names)
            
            # get speaker_ids (tacotron)
            tacotron_speaker_ids = [self.tt_sp_name_lookup[speaker] for speaker in speaker_names]
            tacotron_speaker_ids = torch.LongTensor(tacotron_speaker_ids).cuda()
            
            # get speaker_ids (waveglow)
            waveglow_speaker_ids = self.get_wg_sp_id_from_tt_sp_names(speaker_names)
            waveglow_speaker_ids = [self.wg_sp_id_lookup[int(speaker_id)] for speaker_id in waveglow_speaker_ids]
            waveglow_speaker_ids = torch.LongTensor(waveglow_speaker_ids).cuda()
            
            # get style input
            if style_mode == 'mel':
                mel = load_mel(audio_path.replace(".npy",".wav")).cuda().half()
                style_input = mel
            elif style_mode == 'token':
                pass
                #style_input =
            elif style_mode == 'zeros':
                style_input = None
            elif style_mode == 'torchmoji_hidden':
                try:
                    tokenized, _, _ = self.tm_sentence_tokenizer.tokenize_sentences([text,]) # input array [B] e.g: ["Test?","2nd Sentence!"]
                except:
                    raise Exception(f"text\n{text_batch}\nfailed to tokenize.")
                try:
                    embedding = self.tm_torchmoji(tokenized) # returns np array [B, Embed]
                except Exception as ex:
                    print(f'Exception: {ex}')
                    print(f"text: {text_batch} failed to process.")
                    #raise Exception(f"text\n{text}\nfailed to process.")
                style_input = torch.from_numpy(embedding).cuda().half()
            elif style_mode == 'torchmoji_string':
                style_input = text_batch
                raise NotImplementedError
            else:
                raise NotImplementedError
            
            if not filename:
                filename = str(time.time())
            
            # convert text string to number representation
            sequence = np.array(text_to_sequence(text, self.tt_hparams.text_cleaners))[None, :]
            sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
            
            # run Tacotron
            mel, postmel, gate, alignments, *_ = self.tacotron.inference(sequence, tacotron_speaker_ids, style_input=style_input, style_mode=style_mode)
            
            # run WaveGlow
            audio = self.waveglow.infer(postmel, speaker_ids=waveglow_speaker_ids, sigma=self.wg_train_sigma*0.94)
            audio = audio * 2**15 # scaling for int16 output
            
            # audio = self.denoiser(audio, strength=0.01)[:, 0] # denoise audio output
            
            # move audio to CPU
            audio = audio.squeeze().cpu().numpy().astype('int16')
            
            filename = f"{filename}.wav"
            save_path = os.path.join('server_infer', filename)
            write(save_path, self.tt_hparams.sampling_rate, audio)
            print(f"audio saved at: {save_path}")
        return filename