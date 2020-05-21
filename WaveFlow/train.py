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
from math import e
import math

import numpy as np
import soundfile as sf
from hparams import create_hparams

save_file_check_path = "save"

#=====START: ADDED FOR DISTRIBUTED======
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from torch.utils.data.distributed import DistributedSampler
#from torch.distributed import broadcast
#=====END:   ADDED FOR DISTRIBUTED======

from torch.utils.data import DataLoader
from mel2samp import Mel2Samp
from tqdm import tqdm

# We're using the audio processing from TacoTron2 to make sure it matches
from utils.layers import TacotronSTFT
from utils.utils import StreamingMovingAverage

def get_progress_bar(x, tqdm_params, hparams, rank=0):
    """Returns TQDM iterator if hparams.tqdm, else returns input.
    x = iteration
    tqdm_params = dict of parameters to feed to tqdm
    hparams = A tf.contrib.training.HParams object, containing notebook and tqdm parameters.
    rank = GPU rank, 0 = Main process, greater than 0 = Distributed Processes
    Example usage:
    for i in get_progress_bar(range(0,10), dict(smoothing=0.01), hparams, rank=rank):
        print(i)
    
    Will should progress bar only if hparams.tqdm is True."""
    if rank != 0: # Extra GPU's do not need progress bars
        return x
    
    if hparams.tqdm:
        if hparams.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        return tqdm(x, **tqdm_params)
    else:
        return x


def cprint(*message, b_tqdm=False, **args):
    """Uses tqdm.write() if hparams.notebook else print()"""
    if b_tqdm:
        tqdm.write(" ".join([str(x) for x in message]), *args)
    else:
        print(*message, **args)


def load_checkpoint(warm_start, warm_start_force, checkpoint_path, model, optimizer, scheduler, fp16_run):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    
    if warm_start or warm_start_force:
        iteration = checkpoint_dict['iteration']
        #iteration = 0 # (optional) reset n_iters
    else:
        iteration = checkpoint_dict['iteration']
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    
    checkpoint_model_dict = checkpoint_dict['model']
    if (str(type(model_dict)) != "<class 'collections.OrderedDict'>"):
        checkpoint_model_dict = model_dict.state_dict()
    
    if scheduler and 'scheduler' in checkpoint_dict.keys(): scheduler.load_state_dict(checkpoint_dict['scheduler'])
    if fp16_run and 'amp' in checkpoint_dict.keys(): amp.load_state_dict(checkpoint_dict['amp'])
    
    if warm_start_force:
        model_dict = model.state_dict()
        # Fiter out unneccessary keys
        filtered_dict = {k: v for k,v in checkpoint_model_dict.items() if k in model_dict and checkpoint_model_dict[k].shape == model_dict[k].shape} # shouldn't that be .shape? # yes, that should be .shape
        model_dict_missing = {k: v for k,v in checkpoint_model_dict.items() if k not in model_dict}
        model_dict_mismatching = {k: v for k,v in checkpoint_model_dict.items() if k in model_dict and checkpoint_model_dict[k].shape != model_dict[k].shape}
        pretrained_missing = {k: v for k,v in model_dict.items() if k not in checkpoint_model_dict}
        if model_dict_missing: print(list(model_dict_missing.keys()), "\ndoes not exist in the current model and is being ignored")
        if model_dict_mismatching: print(list(model_dict_mismatching.keys()), "\nis the wrong shape and has been reset")
        if pretrained_missing: print(list(pretrained_missing.keys()), "\ndoesn't have pretrained weights and has been reset")
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
    else:
        checkpoint_model_dict = {k.replace("invconv1x1","convinv").replace(".F.",".WN.").replace("WNs.","WN."): v for k, v in checkpoint_model_dict.items()} # update dictionary as some old checkpoints have out-of-date keynames.
        model.load_state_dict(checkpoint_model_dict)
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model, optimizer, iteration, scheduler


def save_checkpoint(model, optimizer, hparams, learning_rate, iteration, amp, scheduler, speaker_lookup, filepath):
    cprint("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath), b_tqdm=hparams.tqdm)
    saving_dict = {'model': model.state_dict(),
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'learning_rate': learning_rate,
        'speaker_lookup': speaker_lookup,
        'hparams': hparams,
        }
    if amp: saving_dict['amp'] = amp.state_dict()
    torch.save(saving_dict, filepath)
    cprint("Model Saved", b_tqdm=hparams.tqdm)


def validate(model, STFTs, logger, iteration, speaker_lookup, hparams, output_directory, save_audio=True, max_length_s= 5 ):
    from mel2samp import load_wav_to_torch
    val_sigma = hparams.sigma * 0.9
    model.eval()
    with torch.no_grad():
        with open(hparams.validation_files, encoding='utf-8') as f:
            audiopaths_and_melpaths = [line.strip().split('|') for line in f]
        
        if list(model.parameters())[0].type() == "torch.cuda.HalfTensor":
            model_type = "half"
        else:
            model_type = "float"
        
        timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
        total_MAE = total_MSE = total = 0
        for i, (audiopath, melpath, *remaining) in enumerate(audiopaths_and_melpaths):
            if i > 30: break # debug
            audio = load_wav_to_torch(audiopath)[0]/32768.0 # load audio from wav file to tensor
            if audio.shape[0] > (hparams.sampling_rate*max_length_s): continue # ignore audio over max_length_seconds
            mel = np.load(melpath) # load mel from file into numpy arr
            mel = torch.from_numpy(mel).unsqueeze(0).cuda() # from numpy arr to tensor on GPU
            #mel = (mel+5.2)*0.5 # shift values between approx -4 and 4
            if hasattr(model, 'multispeaker') and model.multispeaker == True:
                assert len(remaining), f"Speaker ID missing while multispeaker == True.\nLine: {i}\n'{'|'.join([autiopath, melpath])}'"
                speaker_id = remaining[0]
                assert int(speaker_id) in speaker_lookup.keys(), f"Validation speaker ID:{speaker_id} not found in training filelist.\n(This speaker does not have an embedding, either use single-speaker models or provide an example of this speaker in the training data)."
                speaker_id = torch.IntTensor([speaker_lookup[int(speaker_id)]])
                speaker_id = speaker_id.cuda(non_blocking=True).long()
            else:
                speaker_id = None
            
            if model_type == "half":
                mel = mel.half() # for fp16 training
            
            audio_waveglow = model.infer(mel, speaker_id, sigma=val_sigma)
            audio_waveglow = audio_waveglow.cpu().float()
            
            audio = audio.squeeze().unsqueeze(0) # crush extra dimensions and shape for STFT
            audio_waveglow = audio_waveglow.squeeze().unsqueeze(0) # crush extra dimensions and shape for STFT
            audio_waveglow = audio_waveglow.clamp(-1,1) # clamp any values over/under |1.0| (which should only exist very early in training)
            
            for STFT in STFTs: # check Spectrogram Error with multiple window sizes
                mel_GT = STFT.mel_spectrogram(audio)
                try:
                    mel_waveglow = STFT.mel_spectrogram(audio_waveglow)[:,:,:mel_GT.shape[-1]]
                except AssertionError as ex:
                    cprint(ex, b_tqdm=hparams.tqdm)
                    continue
                
                MSE = (torch.nn.MSELoss()(mel_waveglow, mel_GT)).item() # get MSE (Mean Squared Error) between Ground Truth and WaveGlow inferred spectrograms.
                MAE = (torch.nn.L1Loss()(mel_waveglow, mel_GT)).item() # get MAE (Mean Absolute Error) between Ground Truth and WaveGlow inferred spectrograms.
                
                total_MAE+=MAE
                total_MSE+=MSE
                total+=1
            
            if save_audio:
                audio_path = os.path.join(output_directory, "samples", str(iteration)+"-"+timestr, os.path.basename(audiopath)) # Write audio to checkpoint_directory/iteration/audiofilename.wav
                os.makedirs(os.path.join(output_directory, "samples", str(iteration)+"-"+timestr), exist_ok=True)
                sf.write(audio_path, audio_waveglow.squeeze().cpu().numpy(), hparams.sampling_rate, "PCM_16") # save waveglow sample
                
                audio_path = os.path.join(output_directory, "samples", "Ground Truth", os.path.basename(audiopath)) # Write audio to checkpoint_directory/iteration/audiofilename.wav
                if not os.path.exists(audio_path):
                    os.makedirs(os.path.join(output_directory, "samples", "Ground Truth"), exist_ok=True)
                    sf.write(audio_path, audio.squeeze().cpu().numpy(), hparams.sampling_rate, "PCM_16") # save ground truth
    
    for convinv in model.convinv:
        if hasattr(convinv, 'W_inverse'):
            delattr(convinv, "W_inverse") # clear Inverse Weights.
    
    if total:
        average_MSE = total_MSE/total
        average_MAE = total_MAE/total
        logger.add_scalar('val_MSE', average_MSE, iteration)
        logger.add_scalar('val_MAE', average_MAE, iteration)
        cprint("Average MSE:", average_MSE, "Average MAE:", average_MAE, b_tqdm=hparams.tqdm)
    else:
        average_MSE = 1e3
        average_MAE = 1e3
        cprint("Average MSE: N/A", "Average MAE: N/A", b_tqdm=hparams.tqdm)
    
    model.train()
    return average_MSE, average_MAE


def getCore(hparams):
    core_source = hparams.core_source
    core_type = hparams.core_type
    if "yoyo".lower() in core_source.lower():
        if core_type == "WaveGlow":
            from models.yoyololicon.WaveGlow import WaveGlow
            model = WaveGlow(hparams).cuda()
            from models.yoyololicon.WaveGlowLoss import WaveGlowLoss
            criterion = WaveGlowLoss(hparams)
        elif core_type == "WaveFlow":
            from models.yoyololicon.WaveFlow import WaveFlow
            model = WaveFlow(hparams).cuda()
            from models.L0SG.WaveFlowLoss import WaveFlowLoss
            criterion = WaveFlowLoss(hparams)
        else:
            raise NotImplementedError
    elif "NVIDIA".lower() in core_source.lower():
        from models.NVIDIA.WaveFlow import WaveFlow
        model = WaveFlow(hparams).cuda()
        
        if core_type == "WaveGlow":
            from models.NVIDIA.WaveGlowLoss import WaveGlowLoss
            criterion = WaveGlowLoss(hparams)
        elif core_type == "WaveFlow":
            from models.NVIDIA.WaveFlowLoss import WaveFlowLoss
            criterion = WaveFlowLoss(hparams)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    return model, criterion


def getOptimizer(model, hparams):
    if hparams.optimizer_fused:
        from apex import optimizers as apexopt
        if hparams.optimizer == "Adam":
            optimizer = apexopt.FusedAdam(model.parameters(), lr=hparams.learning_rate)
        elif hparams.optimizer == "LAMB":
            optimizer = apexopt.FusedLAMB(model.parameters(), lr=hparams.learning_rate)
    else:
        if hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate)
        elif hparams.optimizer == "LAMB":
            raise NotImplementedError # PyTorch doesn't currently include LAMB optimizer.
    return optimizer


def train(output_directory, log_directory, checkpoint_path, warm_start, warm_start_force, n_gpus,
          rank, group_name, hparams):
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    #=====START: ADDED FOR DISTRIBUTED======
    if n_gpus > 1:
        init_distributed(rank, n_gpus, group_name, **dist_config)
    #=====END:   ADDED FOR DISTRIBUTED======
    
    model, criterion = getCore(hparams)
    
    #=====START: ADDED FOR DISTRIBUTED======
    if n_gpus > 1:
        model = apply_gradient_allreduce(model)
    #=====END:   ADDED FOR DISTRIBUTED======
    
    STFT = [TacotronSTFT(filter_length=window,
                                 hop_length=hparams.hop_length,
                                 win_length=window,
                                 sampling_rate=hparams.sampling_rate,
                                 n_mel_channels=160,
                                 mel_fmin=hparams.mel_fmin, mel_fmax=hparams.mel_fmax) for window in hparams.validation_windows]
    
    optimizer = getOptimizer(model, hparams)
    
    if hparams.fp16_run:
        global amp
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=hparams.fp16_opt_level, min_loss_scale=2.0)
    else:
        amp = None
    
    # LEARNING RATE SCHEDULER
    if hparams.LRScheduler.lower() == "ReduceLROnPlateau".lower():
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        min_lr = 1e-5
        factor = 0.1**(1/5) # amount to scale the LR by on Validation Loss plateau
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=20, cooldown=2, min_lr=min_lr, verbose=True)
        print("ReduceLROnPlateau used as Learning Rate Scheduler.")
    else: scheduler=None
    
    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path:
        model, optimizer, iteration, scheduler = load_checkpoint(warm_start, warm_start_force, checkpoint_path, model,
                                                      optimizer, scheduler, hparams.fp16_run)
    iteration += 1  # next iteration is iteration + 1
    
    trainset = Mel2Samp(hparams)
    speaker_lookup = trainset.speaker_ids
    # =====START: ADDED FOR DISTRIBUTED======
    if n_gpus > 1:
        train_sampler = DistributedSampler(trainset, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    # =====END:   ADDED FOR DISTRIBUTED======
    train_loader = DataLoader(trainset, num_workers=hparams.n_dataloader_workers, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size,
                              pin_memory=False,
                              drop_last=True)
    
    # Get shared output_directory ready
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)
    
    if rank == 0:
        from tensorboardX import SummaryWriter
        if False: # dated and seperated log dirs for each run
            timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
            log_directory = os.path.join(output_directory, log_directory, timestr)
        else:
            log_directory = os.path.join(output_directory, log_directory)
        logger = SummaryWriter(log_directory)
    
    moving_average = int(min(len(train_loader), 100)) # average loss over 100 iters
    rolling_sum = StreamingMovingAverage(moving_average)
    start_time = time.time()
    start_time_single_batch = time.time()
    
    model.train()
    
    if os.path.exists(os.path.join(output_directory, "best_train_model")):
        best_model_loss = float(str(open(os.path.join(output_directory, "best_train_model")+".txt", "r", encoding="utf-8").read()).split("\n")[0])
    else:
        best_model_loss = -4.20
    if os.path.exists(os.path.join(output_directory, "best_val_model")):
        best_MSE = float(str(open(os.path.join(output_directory, "best_val_model")+".txt", "r", encoding="utf-8").read()).split("\n")[0])
    else:
        best_MSE = 9e9
    epoch_offset = max(0, int(iteration / len(train_loader)))
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("{:,} total parameters.".format(pytorch_total_params))
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("{:,} trainable parameters.".format(pytorch_total_params))
    
    learning_rate = hparams.learning_rate
    # ================ MAIN TRAINING LOOP! ===================
    for epoch in get_progress_bar(range(epoch_offset, hparams.epochs), dict(initial=epoch_offset, total=hparams.epochs, smoothing=0.01, desc="Epoch", position=1, unit="epoch"), hparams, rank=rank):
        cprint(f"Epoch: {epoch}", b_tqdm=hparams.tqdm)
        if n_gpus > 1: train_sampler.set_epoch(epoch)
        
        for i, batch in get_progress_bar(enumerate(train_loader), dict(desc=" Iter", smoothing=0, total=len(train_loader), position=0, unit="iter", leave=True), hparams, rank=rank):
            # run external code every iter, allows the run to be adjusted without restarts
            if (i==0 or iteration % param_interval == 0):
                try:
                    with open("hparams_realtime.py") as f:
                        internal_text = str(f.read())
                        ldict = {'iteration': iteration}
                        exec(internal_text, globals(), ldict)
                except Exception as ex:
                    cprint(f"Custom code FAILED to run!\n{ex}", b_tqdm=hparams.tqdm)
                globals().update(ldict)
                locals().update(ldict)
                if show_live_params:
                    cprint(internal_text, b_tqdm=hparams.tqdm)
            assert warmup_start <= iteration, "Current iteration less than warmup_start."
            # Learning Rate Schedule
            if custom_lr:
                old_lr = learning_rate
                if iteration < warmup_end:
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
                    cprint("Scheduler last_lr overriden. scheduler._last_lr =", scheduler._last_lr, b_tqdm=hparams.tqdm)
                if not iteration % 20: # check actual learning rate every 20 iters (because I sometimes see learning_rate variable go out-of-sync with real LR)
                    learning_rate = optimizer.param_groups[0]['lr']
                if override_scheduler_best:
                    scheduler.best = override_scheduler_best
                    cprint("Scheduler best metric overriden. scheduler.best =", override_scheduler_best, b_tqdm=hparams.tqdm)
            
            model.zero_grad()
            mel, audio, speaker_ids = batch
            mel = torch.autograd.Variable(mel.cuda(non_blocking=True))
            audio = torch.autograd.Variable(audio.cuda(non_blocking=True))
            if model.multispeaker:
                speaker_ids = torch.autograd.Variable(speaker_ids.cuda(non_blocking=True)).long().squeeze(1)
                outputs = model(mel, audio, speaker_ids)
            else:
                outputs = model(mel, audio)
            
            loss = criterion(outputs)
            if n_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()
            
            assert reduced_loss < 1e5, "Model Diverged. Loss > 1e5"
            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
             
            if hparams.b_grad_clip:
                if hparams.fp16_run:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), hparams.grad_clip_thresh)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), hparams.grad_clip_thresh)
                is_overflow = math.isinf(grad_norm) or math.isnan(grad_norm)
            else: is_overflow = False; grad_norm=0.00001
            
            optimizer.step()
            if not is_overflow and rank == 0:
                if (iteration % 100000 == 0):
                    # plot distribution of parameters
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        logger.add_histogram(tag, value.data.cpu().numpy(), iteration)
                logger.add_scalar('training_loss', reduced_loss, iteration)
                if (iteration % 20 == 0):
                    logger.add_scalar('learning.rate', learning_rate, iteration)
                if (iteration % 10 == 0):
                    logger.add_scalar('duration', ((time.time() - start_time)/10), iteration)
                start_time_single_batch = time.time()
            
            average_loss = rolling_sum.process(reduced_loss)
            if rank == 0:
                if (iteration % 10 == 0):
                    cprint("{} {}:  {:.3f}  {:.3f} {:08.3F} {:.8f}LR ({:.8f} Effective)  {:.2f}s/iter {:.4f}s/item".format(time.strftime("%H:%M:%S"), iteration, reduced_loss, average_loss, round(grad_norm,3), learning_rate, min((hparams.grad_clip_thresh/grad_norm)*learning_rate,learning_rate), (time.time() - start_time)/10, ((time.time() - start_time)/10)/(hparams.batch_size*n_gpus)), b_tqdm=hparams.tqdm)
                    start_time = time.time()
                else:
                    cprint("{} {}:  {:.3f}  {:.3f} {:08.3F} {:.8f}LR ({:.8f} Effective)".format(time.strftime("%H:%M:%S"), iteration, reduced_loss, average_loss, round(grad_norm,3), learning_rate, min((hparams.grad_clip_thresh/grad_norm)*learning_rate,learning_rate)), b_tqdm=hparams.tqdm)
            
            if rank == 0 and (len(rolling_sum.values) > moving_average-2):
                if (average_loss+best_model_margin) < best_model_loss:
                    checkpoint_path = os.path.join(output_directory, "best_train_model")
                    try:
                        save_checkpoint(model, optimizer, hparams, learning_rate, iteration, amp, scheduler, speaker_lookup,
                                    checkpoint_path)
                    except KeyboardInterrupt: # Avoid corrupting the model.
                        save_checkpoint(model, optimizer, hparams, learning_rate, iteration, amp, scheduler, speaker_lookup,
                                    checkpoint_path)
                    text_file = open((f"{checkpoint_path}.txt"), "w", encoding="utf-8")
                    text_file.write(str(average_loss)+"\n"+str(iteration))
                    text_file.close()
                    best_model_loss = average_loss #Only save the model if X better than the current loss.
            if rank == 0 and ((iteration % hparams.iters_per_checkpoint == 0) or (os.path.exists(save_file_check_path))):
                checkpoint_path = f"{output_directory}/waveglow_{iteration}"
                save_checkpoint(model, optimizer, hparams, learning_rate, iteration, amp, scheduler, speaker_lookup,
                                checkpoint_path)
                start_time_single_batch = time.time()
                if (os.path.exists(save_file_check_path)):
                    os.remove(save_file_check_path)
            
            if (iteration % validation_interval == 0):
                if rank == 0:
                    MSE, MAE = validate(model, STFT, logger, iteration, speaker_lookup, hparams, output_directory)
                    if scheduler and n_gpus > 1:
                        MSE = torch.tensor(MSE, device='cuda')
                        broadcast(MSE, 0)
                        scheduler.step(MSE.item())
                        if MSE < best_MSE:
                            checkpoint_path = os.path.join(output_directory, "best_val_model")
                            try:
                                save_checkpoint(model, optimizer, hparams, learning_rate, iteration, amp, scheduler, speaker_lookup,
                                            checkpoint_path)
                            except KeyboardInterrupt: # Avoid corrupting the model.
                                save_checkpoint(model, optimizer, hparams, learning_rate, iteration, amp, scheduler, speaker_lookup,
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
            iteration += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str, default="outdir",
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str, default="logdir",
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--warm_start_force', action='store_true',
                        help='load model weights only, ignore all missing/non-matching layers')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    
    args = parser.parse_args()
    hparams = create_hparams(args.hparams)
    
    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)
    
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but only running 1 GPU. Use distributed.py for multiple GPUs")
            n_gpus = 1
    
    if n_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")
    
    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.warm_start_force, n_gpus, args.rank, args.group_name, hparams)

