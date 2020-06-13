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
import argparse
import json
import os
import torch
import os.path
import sys
import time
from math import ceil, e, exp
import math

import numpy as np
import soundfile as sf

save_file_check_path = "save"

#=====START: ADDED FOR DISTRIBUTED======
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import broadcast
#=====END:   ADDED FOR DISTRIBUTED======

from torch.utils.data import DataLoader
from mel2samp import Mel2Samp
from tqdm import tqdm

# We're using the audio processing from TacoTron2 to make sure it matches
sys.path.insert(0, 'tacotron2')
from tacotron2.layers import TacotronSTFT

class LossExplosion(Exception):
    """Custom Exception Class. If Loss Explosion, raise Error and automatically restart training script from previous best_val_model checkpoint."""
    pass

class StreamingMovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []
        self.sum = 0

    def process(self, value):
        self.values.append(value)
        self.sum += value
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        return float(self.sum) / len(self.values)

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, fp16_run, warm_start=False):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    if warm_start:
        if 'iteration' in checkpoint_dict: iteration = checkpoint_dict['iteration']
        #iteration = 0
    else:
        iteration = checkpoint_dict['iteration']
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_dict = checkpoint_dict['model']
    if not warm_start and fp16_run and 'amp' in checkpoint_dict.keys(): amp.load_state_dict(checkpoint_dict['amp'])
    if scheduler and 'scheduler' in checkpoint_dict.keys(): scheduler.load_state_dict(checkpoint_dict['scheduler'])
    
    if warm_start:
        if (str(type(model_dict)) != "<class 'collections.OrderedDict'>"):
            pretrained_dict = model_dict.state_dict() # trained model from checkpoint
        else:
            pretrained_dict = model_dict # trained model from checkpoint
        model_dict = model.state_dict() # just initialized model
        
        # convert Nvidia/WaveGlow into my format
        if 0:
            pretrained_dict = {k.replace(".in_layers",".WN.in_layers").replace(".res_skip_layers",".WN.res_skip_layers").replace(".start",".WN.start").replace(".end",".WN.end").replace(".conv.weight",".weight"): v for k, v in pretrained_dict.items()}
        
        # Fiter out unneccessary keys
        filtered_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict and pretrained_dict[k].shape == model_dict[k].shape} # shouldn't that be .shape?
        model_dict_missing = {k: v for k,v in pretrained_dict.items() if k not in model_dict}
        model_dict_mismatching = {k: v for k,v in pretrained_dict.items() if k in model_dict and pretrained_dict[k].shape != model_dict[k].shape}
        pretrained_missing = {k: v for k,v in model_dict.items() if k not in pretrained_dict}
        if model_dict_missing: print(list(model_dict_missing.keys()),'does not exist in the current model and is being ignored\n\n')
        if model_dict_mismatching: print(list(model_dict_mismatching.keys()),"is the wrong shape and has been reset\n\n")
        if pretrained_missing: print(list(pretrained_missing.keys()),"doesn't have pretrained weights and has been reset\n\n")
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
    else:
        if (str(type(model_dict)) != "<class 'collections.OrderedDict'>"):
            model_dict = model_dict.state_dict()
        model_dict = {k.replace("invconv1x1","convinv").replace(".F.",".WN.").replace("WNs.","WN."): v for k, v in model_dict.items()}
        model.load_state_dict(model_dict)
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model, optimizer, iteration, scheduler

def save_checkpoint(model, optimizer, learning_rate, iteration, amp, scheduler, speaker_lookup, filepath):
    tqdm.write("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))
    #model_for_saving = WaveGlow(**waveglow_config).cuda()
    #model_for_saving.load_state_dict(model.state_dict())
    #state_dict = model_for_saving.state_dict()
    state_dict = model.state_dict()
    saving_dict = {'model': state_dict,
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'learning_rate': learning_rate,
        'speaker_lookup': speaker_lookup,
        'waveglow_config': waveglow_config,
        }
    if amp: saving_dict['amp'] = amp.state_dict()
    torch.save(saving_dict, filepath)
    tqdm.write("Model Saved")

def save_weights(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model weights without optimizer at iteration {} to {}".format(
          iteration, filepath))
    model_for_saving = WaveGlow(**waveglow_config).cuda()
    model_for_saving.load_state_dict(model.state_dict())
    #torch.save(model_for_saving.state_dict(), filepath)
    torch.save({'model': model_for_saving,
                'iteration': iteration}, filepath)
    print("Weights Saved")

def validate(model, loader_STFT, STFTs, logger, iteration, validation_files, speaker_lookup, sigma, output_directory, data_config, save_audio=True, max_length_s= 3 ):
    from mel2samp import load_wav_to_torch
    from scipy import signal
    val_sigma = sigma * 0.95
    model.eval()
    with torch.no_grad():
        with open(validation_files, encoding='utf-8') as f:
            audiopaths_and_melpaths = [line.strip().split('|') for line in f]
        
        if list(model.parameters())[0].type() == "torch.cuda.HalfTensor":
            model_type = "half"
        else:
            model_type = "float"
        
        timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
        total_MAE = total_MSE = total = files_processed = 0
        for i, (audiopath, melpath, *remaining) in enumerate(audiopaths_and_melpaths):
            if files_processed >= 7: # number of validation files to run.
                break
            audio = load_wav_to_torch(audiopath)[0]/32768.0 # load audio from wav file to tensor
            if audio.shape[0] > (data_config['sampling_rate']*max_length_s): continue # ignore audio over max_length_seconds
            
            if loader_STFT:
                mel = loader_STFT.mel_spectrogram(audio.unsqueeze(0)).cuda()
            else:
                mel = np.load(melpath) # load mel from file into numpy arr
                mel = torch.from_numpy(mel).unsqueeze(0).cuda() # from numpy arr to tensor on GPU
            
            if hasattr(model, 'multispeaker') and model.multispeaker == True:
                assert len(remaining), f"Speaker ID missing while multispeaker == True.\nLine: {i}\n'{'|'.join([autiopath, melpath])}'"
                speaker_id = remaining[0]
                speaker_id = torch.IntTensor([speaker_lookup[int(speaker_id)]])
                speaker_id = speaker_id.cuda(non_blocking=True).long()
            else:
                speaker_id = None
            
            if model_type == "half":
                mel = mel.half() # for fp16 training
            
            audio_waveglow = model.infer(mel, speaker_id, sigma=val_sigma).cpu().float()
            
            if data_config['preempthasis']: # inverse-preempthasis
                audio_waveglow = audio_waveglow.squeeze()
                audio_waveglow = torch.from_numpy(signal.lfilter([1], [1, -float(data_config['preempthasis'])], audio_waveglow.numpy())).float() # de-preempthasis (scipy signal is faster than pytorch implementation for some reason /shrug )
            
            audio = audio.squeeze().unsqueeze(0) # crush extra dimensions and shape for STFT
            audio_waveglow = audio_waveglow.squeeze().unsqueeze(0) # crush extra dimensions and shape for STFT
            audio_waveglow = audio_waveglow.clamp(-1,1) # clamp any values over/under |1.0| (which should only exist very early in training)
            
            for STFT in STFTs: # check Spectrogram Error with multiple window sizes
                mel_GT = STFT.mel_spectrogram(audio)
                try:
                    mel_waveglow = STFT.mel_spectrogram(audio_waveglow)[:,:,:mel_GT.shape[-1]]
                except AssertionError as ex:
                    print(ex)
                    continue
                
                MSE = (torch.nn.MSELoss()(mel_waveglow, mel_GT)).item() # get MSE (Mean Squared Error) between Ground Truth and WaveGlow inferred spectrograms.
                MAE = (torch.nn.L1Loss()(mel_waveglow, mel_GT)).item() # get MAE (Mean Absolute Error) between Ground Truth and WaveGlow inferred spectrograms.
                
                total_MAE+=MAE
                total_MSE+=MSE
                total+=1
            
            if save_audio:
                audio_path = os.path.join(output_directory, "samples", str(iteration)+"-"+timestr, os.path.basename(audiopath)) # Write audio to checkpoint_directory/iteration/audiofilename.wav
                os.makedirs(os.path.join(output_directory, "samples", str(iteration)+"-"+timestr), exist_ok=True)
                sf.write(audio_path, audio_waveglow.squeeze().cpu().numpy(), data_config['sampling_rate'], "PCM_16") # save waveglow sample
                
                audio_path = os.path.join(output_directory, "samples", "Ground Truth", os.path.basename(audiopath)) # Write audio to checkpoint_directory/iteration/audiofilename.wav
                if not os.path.exists(audio_path):
                    os.makedirs(os.path.join(output_directory, "samples", "Ground Truth"), exist_ok=True)
                    sf.write(audio_path, audio.squeeze().cpu().numpy(), data_config['sampling_rate'], "PCM_16") # save ground truth
            files_processed+=1
    
    for convinv in model.convinv:
        if hasattr(convinv, 'W_inverse'):
            delattr(convinv, "W_inverse") # clear Inverse Weights.
    
    del mel, speaker_id # del GPU based tensors.
    
    torch.cuda.empty_cache() # clear cache for next training
    
    if total:
        average_MSE = total_MSE/total
        average_MAE = total_MAE/total
        logger.add_scalar('val_MSE', average_MSE, iteration)
        logger.add_scalar('val_MAE', average_MAE, iteration)
        print("Average MSE:", average_MSE, "Average MAE:", average_MAE)
    else:
        average_MSE = 1e3
        average_MAE = 1e3
        print("Average MSE: N/A", "Average MAE: N/A")
    model.train()
    return average_MSE, average_MAE

def multiLR(model):
    model_parameters = []
    model_parameters.append( {"params": list(model.children())[0].parameters()} ) # ConvTranspose1d - Mel upscaling layer
    model_parameters.append( {"params": list(model.children())[2].parameters()} ) # Invertible1x1Conv - ConvInv
    
    # [0, 1]
    # [2:2+in_layers_len]
    
    in_layers_ = [ {"params": x.in_layers.parameters()} for x in list(model.children())[1] ]
    in_layers_offsets = [2, 2+len(in_layers_)]
    model_parameters.extend( in_layers_ ) # in layers
    
    res_skip_layers_ = [ {"params": x.res_skip_layers.parameters()} for x in list(model.children())[1] ]
    res_skip_layers_offsets = [in_layers_offsets[1], in_layers_offsets[1]+len(res_skip_layers_)]
    model_parameters.extend( res_skip_layers_ ) # residual and skip layers
    
    start_ = [ {"params": x.start.parameters()} for x in list(model.children())[1] ]
    start_offsets = [res_skip_layers_offsets[1], res_skip_layers_offsets[1]+len(start_)]
    model_parameters.extend( start_ ) # start layers
    
    end_ = [ {"params": x.end.parameters()} for x in list(model.children())[1] ]
    end_offsets = [start_offsets[1], start_offsets[1]+len(end_)]
    model_parameters.extend( end_ ) # end layers
    
    cond_layer_ = [ {"params": x.cond_layer.parameters()} for x in list(model.children())[1] ]
    cond_layer_offsets = [end_offsets[1], end_offsets[1]+len(cond_layer_)]
    model_parameters.extend( cond_layer_ ) # conditioning layers
    return model_parameters, (in_layers_offsets, res_skip_layers_offsets, start_offsets, end_offsets, cond_layer_offsets)
    
def train(num_gpus, rank, group_name, output_directory, epochs, learning_rate,
          sigma, loss_empthasis, iters_per_checkpoint, batch_size, seed, fp16_run,
          checkpoint_path, with_tensorboard, logdirname, datedlogdir, warm_start=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)
    #=====END:   ADDED FOR DISTRIBUTED======
    
    global WaveGlow
    global WaveGlowLoss
    
    ax = True # this is **really** bad coding practice :D
    if ax:
        from efficient_model_ax import WaveGlow
        from efficient_loss import WaveGlowLoss
    else:
        if waveglow_config["yoyo"]: # efficient_mode # TODO: Add to Config File
            from efficient_model import WaveGlow
            from efficient_loss import WaveGlowLoss
        else:
            from glow import WaveGlow, WaveGlowLoss
    
    criterion = WaveGlowLoss(sigma, loss_empthasis)
    model = WaveGlow(**waveglow_config).cuda()
    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)
    #=====END:   ADDED FOR DISTRIBUTED======
    STFT = [TacotronSTFT(filter_length=window,
                                 hop_length=data_config['hop_length'],
                                 win_length=window,
                                 sampling_rate=data_config['sampling_rate'],
                                 n_mel_channels=160,
                                 mel_fmin=0, mel_fmax=16000) for window in data_config['validation_windows']]
    
    loader_STFT = TacotronSTFT(filter_length=data_config['filter_length'],
                                 hop_length=data_config['hop_length'],
                                 win_length=data_config['win_length'],
                                 sampling_rate=data_config['sampling_rate'],
                                 n_mel_channels=160,
                                 mel_fmin=data_config['mel_fmin'], mel_fmax=data_config['mel_fmax'])
    
    optimizer = "Adam"
    optimizer_fused = True # use Apex fused optimizer, should be identical to normal but slightly faster
    if optimizer_fused:
        from apex import optimizers as apexopt
        if optimizer == "Adam":
            optimizer = apexopt.FusedAdam(model.parameters(), lr=learning_rate)
        elif optimizer == "LAMB":
            optimizer = apexopt.FusedLAMB(model.parameters(), lr=learning_rate, max_grad_norm=1e6)
    else:
        if optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer == "LAMB":
            from lamb import Lamb as optLAMB
            optimizer = optLAMB(model.parameters(), lr=learning_rate)
            #import torch_optimizer as optim
            #optimizer = optim.Lamb(model.parameters(), lr=learning_rate)
            #raise# PyTorch doesn't currently include LAMB optimizer.
    
    if fp16_run:
        global amp
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    else:
        amp = None
    
    ## LEARNING RATE SCHEDULER
    if True:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        min_lr = 1e-5
        factor = 0.1**(1/5) # amount to scale the LR by on Validation Loss plateau
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=20, cooldown=2, min_lr=min_lr, verbose=True)
        print("ReduceLROnPlateau used as Learning Rate Scheduler.")
    else: scheduler=False
    
    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path != "":
        
        #warm_start = 0 # WARM START THE MODEL AND RESET ANY INVALID LAYERS
        
        model, optimizer, iteration, scheduler = load_checkpoint(checkpoint_path, model,
                                                      optimizer, scheduler, fp16_run, warm_start=warm_start)
        iteration += 1  # next iteration is iteration + 1
    
    trainset = Mel2Samp(**data_config, check_files=True)
    speaker_lookup = trainset.speaker_ids
    # =====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        train_sampler = DistributedSampler(trainset, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    # =====END:   ADDED FOR DISTRIBUTED======
    train_loader = DataLoader(trainset, num_workers=2, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)
    
    # Get shared output_directory ready
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)
    
    if with_tensorboard and rank == 0:
        from tensorboardX import SummaryWriter
        if datedlogdir:
            timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
            log_directory = os.path.join(output_directory, logdirname, timestr)
        else:
            log_directory = os.path.join(output_directory, logdirname)
        logger = SummaryWriter(log_directory)
    
    moving_average = int(min(len(train_loader), 100)) # average loss over entire Epoch
    rolling_sum = StreamingMovingAverage(moving_average)
    start_time = time.time()
    start_time_single_batch = time.time()
    model.train()
    
    # best (averaged) training loss
    if os.path.exists(os.path.join(output_directory, "best_model")+".txt"):
        best_model_loss = float(str(open(os.path.join(output_directory, "best_model")+".txt", "r", encoding="utf-8").read()).split("\n")[0])
    else:
        best_model_loss = -6.20
    
    # best (validation) MSE on inferred spectrogram.
    if os.path.exists(os.path.join(output_directory, "best_val_model")+".txt"):
        best_MSE = float(str(open(os.path.join(output_directory, "best_val_model")+".txt", "r", encoding="utf-8").read()).split("\n")[0])
    else:
        best_MSE = 9e9
    
    epoch_offset = max(0, int(iteration / len(train_loader)))
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("{:,} total parameters in model".format(pytorch_total_params))
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("{:,} trainable parameters.".format(pytorch_total_params))
    
    training = True
    while training:
        try:
            if rank == 0:
                epochs_iterator = tqdm(range(epoch_offset, epochs), initial=epoch_offset, total=epochs, smoothing=0.01, desc="Epoch", position=1, unit="epoch")
            else:
                epochs_iterator = range(epoch_offset, epochs)
            # ================ MAIN TRAINING LOOP! ===================
            for epoch in epochs_iterator:
                print(f"Epoch: {epoch}")
                if num_gpus > 1:
                    train_sampler.set_epoch(epoch)
                
                if rank == 0:
                    iters_iterator = tqdm(enumerate(train_loader), desc=" Iter", smoothing=0, total=len(train_loader), position=0, unit="iter", leave=True)
                else:
                    iters_iterator = enumerate(train_loader)
                for i, batch in iters_iterator:
                    # run external code every iter, allows the run to be adjusted without restarts
                    if (i==0 or iteration % param_interval == 0):
                        try:
                            with open("run_every_epoch.py") as f:
                                internal_text = str(f.read())
                                if len(internal_text) > 0:
                                    #code = compile(internal_text, "run_every_epoch.py", 'exec')
                                    ldict = {'iteration': iteration}
                                    exec(internal_text, globals(), ldict)
                                else:
                                    print("No Custom code found, continuing without changes.")
                        except Exception as ex:
                            print(f"Custom code FAILED to run!\n{ex}")
                        globals().update(ldict)
                        locals().update(ldict)
                        if show_live_params:
                            print(internal_text)
                    if not iteration % 50: # check actual learning rate every 20 iters (because I sometimes see learning_rate variable go out-of-sync with real LR)
                        learning_rate = optimizer.param_groups[0]['lr']
                    # Learning Rate Schedule
                    if custom_lr:
                        old_lr = learning_rate
                        if iteration < warmup_start:
                            learning_rate = warmup_start_lr
                        elif iteration < warmup_end:
                            learning_rate = (iteration-warmup_start)*((A_+C_)-warmup_start_lr)/(warmup_end-warmup_start) + warmup_start_lr # learning rate increases from warmup_start_lr to A_ linearly over (warmup_end-warmup_start) iterations.
                        else:
                            if iteration < decay_start:
                                learning_rate = A_ + C_
                            else:
                                iteration_adjusted = iteration - decay_start
                                learning_rate = (A_*(e**(-iteration_adjusted/B_))) + C_
                        assert learning_rate > -1e-8, "Negative Learning Rate."
                        if old_lr != learning_rate:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = learning_rate
                    else:
                        scheduler.patience = scheduler_patience
                        scheduler.cooldown = scheduler_cooldown
                        if override_scheduler_last_lr:
                            scheduler._last_lr = override_scheduler_last_lr
                            print("Scheduler last_lr overriden. scheduler._last_lr =", scheduler._last_lr)
                        if override_scheduler_best:
                            scheduler.best = override_scheduler_best
                            print("Scheduler best metric overriden. scheduler.best =", override_scheduler_best)
                    model.zero_grad()
                    mel, audio, speaker_ids = batch
                    mel = torch.autograd.Variable(mel.cuda(non_blocking=True))
                    audio = torch.autograd.Variable(audio.cuda(non_blocking=True))
                    if waveglow_config['WN_config']['speaker_embed_dim'] > 0:
                        speaker_ids = speaker_ids.cuda(non_blocking=True).long().squeeze(1)
                        outputs = model(mel, audio, speaker_ids)
                    else:
                        outputs = model(mel, audio, None)
                    
                    loss = criterion(outputs)
                    if num_gpus > 1:
                        reduced_loss = reduce_tensor(loss.data, num_gpus).item()
                    else:
                        reduced_loss = loss.item()
                    
                    if fp16_run:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    
                    if iteration > 1e3 and ((reduced_loss > LossExplosionThreshold) or (math.isnan(reduced_loss))):
                        model.zero_grad()
                        raise LossExplosion(f"\nLOSS EXPLOSION EXCEPTION ON RANK {rank}: Loss reached {reduced_loss} during iteration {iteration}.\n\n\n")
                    
                    grad_clip = True; grad_clip_thresh = 500
                    if grad_clip:
                        if fp16_run:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                amp.master_params(optimizer), grad_clip_thresh)
                        else:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(), grad_clip_thresh)
                        is_overflow = math.isinf(grad_norm) or math.isnan(grad_norm)
                    else: is_overflow = False; grad_norm=0.00001
                    
                    optimizer.step()
                    if not is_overflow and with_tensorboard and rank == 0:
                        if (iteration % 100000 == 0):
                            # plot distribution of parameters
                            for tag, value in model.named_parameters():
                                tag = tag.replace('.', '/')
                                logger.add_histogram(tag, value.data.cpu().numpy(), iteration)
                        logger.add_scalar('training_loss', reduced_loss, iteration)
                        #logger.add_scalar('training_loss_exp', 500*(exp(reduced_loss)), iteration)
                        logger.add_scalar('training_loss_samples', reduced_loss, iteration*batch_size)
                        if (iteration % 20 == 0):
                            logger.add_scalar('learning.rate', learning_rate, iteration)
                        if (iteration % 10 == 0):
                            logger.add_scalar('duration', ((time.time() - start_time)/10), iteration)
                        start_time_single_batch = time.time()
                    
                    average_loss = rolling_sum.process(reduced_loss)
                    if rank == 0:
                        if (iteration % 10 == 0):
                            tqdm.write("{} {}:  {:.3f}  {:.3f} {:08.3F} {:.8f}LR ({:.8f} Effective)  {:.2f}s/iter {:.4f}s/item".format(time.strftime("%H:%M:%S"), iteration, reduced_loss, average_loss, round(grad_norm,3), learning_rate, min((grad_clip_thresh/grad_norm)*learning_rate,learning_rate), (time.time() - start_time)/10, ((time.time() - start_time)/10)/(batch_size*num_gpus)))
                            start_time = time.time()
                        else:
                            tqdm.write("{} {}:  {:.3f}  {:.3f} {:08.3F} {:.8f}LR ({:.8f} Effective)".format(time.strftime("%H:%M:%S"), iteration, reduced_loss, average_loss, round(grad_norm,3), learning_rate, min((grad_clip_thresh/grad_norm)*learning_rate,learning_rate)))
                    
                    if rank == 0 and (len(rolling_sum.values) > moving_average-2):
                        if (average_loss+best_model_margin) < best_model_loss:
                            checkpoint_path = os.path.join(output_directory, "best_model")
                            try:
                                save_checkpoint(model, optimizer, learning_rate, iteration, amp, scheduler, speaker_lookup,
                                            checkpoint_path)
                            except KeyboardInterrupt: # Avoid corrupting the model.
                                save_checkpoint(model, optimizer, learning_rate, iteration, amp, scheduler, speaker_lookup,
                                            checkpoint_path)
                            text_file = open((f"{checkpoint_path}.txt"), "w", encoding="utf-8")
                            text_file.write(str(average_loss)+"\n"+str(iteration))
                            text_file.close()
                            best_model_loss = average_loss #Only save the model if X better than the current loss.
                    if rank == 0 and ((iteration % iters_per_checkpoint == 0) or (os.path.exists(save_file_check_path))):
                        checkpoint_path = f"{output_directory}/waveglow_{iteration}"
                        save_checkpoint(model, optimizer, learning_rate, iteration, amp, scheduler, speaker_lookup,
                                        checkpoint_path)
                        start_time_single_batch = time.time()
                        if (os.path.exists(save_file_check_path)):
                            os.remove(save_file_check_path)
                    
                    if (iteration % validation_interval == 0):
                        if rank == 0:
                            MSE, MAE = validate(model, loader_STFT, STFT, logger, iteration, data_config['validation_files'], speaker_lookup, sigma, output_directory, data_config)
                            if scheduler:
                                MSE = torch.tensor(MSE, device='cuda')
                                if num_gpus > 1:
                                    broadcast(MSE, 0)
                                scheduler.step(MSE.item())
                                if MSE < best_MSE:
                                    checkpoint_path = os.path.join(output_directory, "best_val_model")
                                    try:
                                        save_checkpoint(model, optimizer, learning_rate, iteration, amp, scheduler, speaker_lookup,
                                                    checkpoint_path)
                                    except KeyboardInterrupt: # Avoid corrupting the model.
                                        save_checkpoint(model, optimizer, learning_rate, iteration, amp, scheduler, speaker_lookup,
                                                    checkpoint_path)
                                    text_file = open((f"{checkpoint_path}.txt"), "w", encoding="utf-8")
                                    text_file.write(str(MSE.item())+"\n"+str(iteration))
                                    text_file.close()
                                    best_MSE = MSE.item() #Only save the model if X better than the current loss.
                        else:
                            if scheduler:
                                MSE = torch.zeros(1, device='cuda')
                                broadcast(MSE, 0)
                                scheduler.step(MSE.item())
                        learning_rate = optimizer.param_groups[0]['lr'] #check actual learning rate (because I sometimes see learning_rate variable go out-of-sync with real LR)
                    iteration += 1
            training = False # exit the While loop
        
        except LossExplosion as ex: # print Exception and continue from checkpoint. (turns out it takes < 4 seconds to restart like this, fucking awesome)
            print(ex) # print Loss
            checkpoint_path = os.path.join(output_directory, "best_val_model")
            assert os.path.exists(checkpoint_path), "best_val_model must exist for automatic restarts"
            
            # clearing VRAM for load checkpoint
            audio = mel = speaker_ids = loss = None
            torch.cuda.empty_cache()
            
            model.eval()
            model, optimizer, iteration, scheduler = load_checkpoint(checkpoint_path, model, optimizer, scheduler, fp16_run)
            learning_rate = optimizer.param_groups[0]['lr']
            epoch_offset = max(0, int(iteration / len(train_loader)))
            model.train()
            iteration += 1
            pass # and continue training.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    args = parser.parse_args()
    
    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
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
    
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU.  Use distributed.py for multiple GPUs")
            num_gpus = 1
    
    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train(num_gpus, args.rank, args.group_name, **train_config)
