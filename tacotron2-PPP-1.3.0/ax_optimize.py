import os
import time
import argparse
import math
import numpy as np
from numpy import finfo

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import load_model
from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams
from utils import to_gpu
import time
from math import e

from tqdm import tqdm
import layers
from utils import load_wav_to_torch
from scipy.io.wavfile import read

import os.path

from metric import alignment_metric
import concurrent.futures


save_file_check_path = "save"
num_workers_ = 2
start_from_checkpoints_from_zero = 0
gen_new_mels = 0


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


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams,
                           speaker_ids=trainset.speaker_ids)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset,shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=num_workers_, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False, # default pin_memory=False, True should allow async memory transfers # Causes very random CUDA errors (after like 4+ hours)
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn, train_sampler


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    #optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    try: best_validation_loss = checkpoint_dict['best_validation_loss']; print("best_validation_loss found in checkpoint\nbest_validation_loss: ",best_validation_loss)
    except: print("best_validation_loss not found in checkpoint, using default")
    if (start_from_checkpoints_from_zero):
        iteration = 0
    else:
        iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration, best_validation_loss


def save_checkpoint(model, optimizer, learning_rate, iteration, best_validation_loss, filepath):
    tqdm.write("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate,
                'best_validation_loss': best_validation_loss}, filepath)
    tqdm.write("Saving Complete")


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=num_workers_,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, drop_last=True, collate_fn=collate_fn)

        val_loss = 0.0
        diagonality = torch.zeros(1)
        avg_prob = torch.zeros(1)
        for i, batch in tqdm(enumerate(val_loader), desc="Validation", total=len(val_loader), smoothing=0): # i = index, batch = stuff in array[i]
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            # text_padded, input_lengths, mel_padded, max_len, output_lengths, speaker_ids = x
            # mel_out, mel_out_postnet, gate_outputs, alignments = y_pred
            rate, prob = alignment_metric(x, y_pred)
            diagonality += rate
            avg_prob += prob
            loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
            # end forloop
        val_loss = val_loss / (i + 1)
        diagonality = (diagonality / (i + 1)).item()
        avg_prob = (avg_prob / (i + 1)).item()
        # end torch.no_grad()

    model.train()
    if rank == 0:
        tqdm.write("Validation loss {}: {:9f}  Average Max Attention: {:9f}".format(iteration, val_loss, avg_prob))
        #logger.log_validation(val_loss, model, y, y_pred, iteration)
        if iteration != 0:
            logger.log_validation(val_loss, model, y, y_pred, iteration, diagonality, avg_prob)
    return val_loss, avg_prob


def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams, ax_max_run_timer, parameters):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    train_loader, valset, collate_fn, train_sampler = prepare_dataloaders(hparams)

    model = load_model(hparams)
    model.train()
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2', min_loss_scale=1.0)#, loss_scale=256.0)

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss()
    
    log_dir_counter = 0
    starting_log_directory = log_directory
    while os.path.exists(log_directory):
        log_dir_counter+=1
        log_directory = starting_log_directory+"_"+str(log_dir_counter)
    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    # Load checkpoint if one exists
    best_validation_loss = 0.6 # used to see when "best_model" should be saved, default = 0.4, load_checkpoint will update to last best value.
    val_avg_prob = 0.0
    iteration = 0
    epoch_offset = 0
    _learning_rate = 1e-3
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration, best_validation_loss = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))
    # define scheduler
    use_scheduler = 0
    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, factor=0.562341325, patience=15)

    model.train()
    is_overflow = False
    validate_then_terminate = 0
    if validate_then_terminate:
        val_loss = validate(model, criterion, valset, iteration,
            hparams.batch_size, n_gpus, collate_fn, logger,
            hparams.distributed_run, rank)
        raise Exception("Finished Validation")
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    decay_start = parameters["decay_start"]
    A_ = parameters["lr_A"]
    B_ = parameters["lr_B"]
    C_ = 0
    min_learning_rate = 1e-6
    epochs_between_updates = parameters["epochs_between_updates"]
    p_teacher_forcing = 1.00
    teacher_force_till = 30
    rolling_loss = StreamingMovingAverage(int(len(train_loader)))
    ax_start_time = time.time()
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in tqdm(range(epoch_offset, hparams.epochs), initial=epoch_offset, total=hparams.epochs, desc="Epoch:", position=1, unit="epoch"):
        tqdm.write("Epoch:{}".format(epoch))
        
        # run external code every epoch, allows the run to be adjusting without restarts
        try:
            with open("run_every_epoch.py") as f:
                internal_text = str(f.read())
                if len(internal_text) > 0:
                    print(internal_text)
                    #code = compile(internal_text, "run_every_epoch.py", 'exec')
                    ldict = {}
                    exec(internal_text, globals(), ldict)
                    C_ = ldict['C_']
                    min_learning_rate = ldict['min_learning_rate']
                    p_teacher_forcing = ldict['p_teacher_forcing']
                    teacher_force_till = ldict['teacher_force_till']
                    print("Custom code excecuted\nPlease remove code if it was intended to be ran once.")
                else:
                    print("No Custom code found, continuing without changes.")
        except Exception as ex:
            print(f"Custom code FAILED to run!\n{ex}")
        print("decay_start is ",decay_start)
        print("A_ is ",A_)
        print("B_ is ",B_)
        print("C_ is ",C_)
        print("min_learning_rate is ",min_learning_rate)
        print("epochs_between_updates is ",epochs_between_updates)
        print("p_teacher_forcing is ",p_teacher_forcing)
        print("teacher_force_till is ",teacher_force_till)
        if epoch % epochs_between_updates == 0 or epoch_offset == epoch:
        #if None:
            tqdm.write("Old learning rate [{:.6f}]".format(learning_rate))
            if iteration < decay_start:
                learning_rate = A_
            else:
                iteration_adjusted = iteration - decay_start
                learning_rate = (A_*(e**(-iteration_adjusted/B_))) + C_
            learning_rate = max(min_learning_rate, learning_rate) # output the largest number
            #if epoch_offset == epoch: # hold learning rate low during first pass to let optimizer rebuild
            #    learning_rate = 1e-5
            tqdm.write("Changing Learning Rate to [{:.6f}]".format(learning_rate))
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        
        if hparams.distributed_run: # shuffles the train_loader when doing multi-gpu training
            train_sampler.set_epoch(epoch)
        start_time = time.time()
        # start iterating through the epoch
        for i, batch in tqdm(enumerate(train_loader), desc="Iter:  ", smoothing=0, total=len(train_loader), position=0, unit="iter"):
            model.zero_grad()
            x, y = model.parse_batch(batch) # move batch to GPU (async)
            y_pred = model(x, teacher_force_till=teacher_force_till, p_teacher_forcing=p_teacher_forcing)
            
            loss = criterion(y_pred, y)
            
            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()

            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)
            
            optimizer.step()
            
            for j, param_group in enumerate(optimizer.param_groups):
                learning_rate = (float(param_group['lr'])); break
            
            if not is_overflow and rank == 0:
                duration = time.time() - start_time
                average_loss = rolling_loss.process(reduced_loss)
                tqdm.write("{} [Train_loss {:.4f} Avg {:.4f}] [Grad Norm {:.4f}] "
                      "[{:.2f}s/it] [{:.3f}s/file] [{:.7f} LR]".format(
                    iteration, reduced_loss, average_loss, grad_norm, duration, (duration/(hparams.batch_size*n_gpus)), learning_rate))
                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration)
                start_time = time.time()
            if is_overflow and rank == 0:
                tqdm.write("Gradient Overflow, Skipping Step")

                if rank == 0:
                    if (os.path.exists(save_file_check_path)):
                        os.remove(save_file_check_path)
            
            if (time.time() - ax_start_time) > ax_max_run_timer:
                break
            iteration += 1
            # end of iteration loop
        # end of epoch loop
        # perform validation and save "ax_model"
        val_loss, val_avg_prob = validate(model, criterion, valset, iteration,
                 hparams.batch_size, n_gpus, collate_fn, logger,
                 hparams.distributed_run, rank)
        if use_scheduler:
            scheduler.step(val_loss)
        if rank == 0:
            checkpoint_path = os.path.join(output_directory, "ax_model")
            save_checkpoint(model, optimizer, learning_rate, iteration,
                        best_validation_loss, checkpoint_path)
        
        # lets pretend this code is actually able to finish
        return val_avg_prob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
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
    
#-------------------------------- OPTIMIZE PART ------------------------------------
    import ax
    from ax import optimize
    def train_experiment_thread(parameters):
        print(str(parameters)[2:-2].replace(", '","\n").replace("': "," = "))
        # {'decay_start': 4730, 'lr_A': 0.0004209955514847238, 'lr_B': 761.0139550594057, 'epochs_between_updates': 1, 'decoder_rnn_dim': 1090, 'p_attention_dropout': 0.11175302416086197, 'p_decoder_dropout': 0.4907491207122803, 'attention_rnn_dim': 3563, 'attention_dim': 505, 'attention_location_n_filters': 114, 'speaker_embedding_dim': 905}
        
        
        # {'decay_start': 20796.121660564026
        #lr_A': 0.002119413859083535
        #lr_B': 1851.3716159865699
        #epochs_between_updates': 1
        #decoder_rnn_dim': 984
        #p_attention_dropout': 0.2142794519662857
        #p_decoder_dropout': 0.0028651717118918897
        #attention_rnn_dim': 1740
        #attention_dim': 165
        #attention_location_n_filters': 66
        #speaker_embedding_dim': 1601}
        
        hparams.decoder_rnn_dim = parameters["decoder_rnn_dim"]
        hparams.p_attention_dropout = parameters["p_attention_dropout"]
        hparams.p_decoder_dropout = parameters["p_decoder_dropout"]
        hparams.attention_rnn_dim = parameters["attention_rnn_dim"]
        hparams.attention_dim = parameters["attention_dim"]
        hparams.attention_location_n_filters = parameters["attention_location_n_filters"]
        hparams.speaker_embedding_dim = parameters["speaker_embedding_dim"]
        
        # ---------------------- CUSTOM CONFIG ------------------------------------
        ax_max_run_timer_minutes = 60 # timer in minutes
        hparams.batch_size = 64 # batch size for ax opimizing
        hparams.ignore_layers=["decoder.attention_rnn.weight_ih","decoder.attention_rnn.weight_hh","decoder.attention_rnn.bias_ih","decoder.attention_rnn.bias_hh","decoder.attention_layer.query_layer.linear_layer.weight","decoder.attention_layer.memory_layer.linear_layer.weight","decoder.attention_layer.v.linear_layer.weight","decoder.attention_layer.location_layer.location_conv.conv.weight","decoder.attention_layer.location_layer.location_dense.linear_layer.weight","decoder.decoder_rnn.weight_ih","decoder.decoder_rnn.weight_hh","decoder.decoder_rnn.bias_ih","decoder.decoder_rnn.bias_hh","decoder.linear_projection.linear_layer.weight","decoder.gate_layer.linear_layer.weight","speaker_embedding.weight"]
        
        start_timer = time.time()
        ax_max_run_timer_seconds = 60*ax_max_run_timer_minutes
        try:
            val_avg_prob = train(args.output_directory, args.log_directory, args.checkpoint_path,
                  args.warm_start, args.n_gpus, args.rank, args.group_name, hparams, ax_max_run_timer_seconds, parameters)
        except Exception as ex:
            torch.cuda.empty_cache()
            print("Exception Occured:\n",ex,"\nSetting score to lower bound\n")
            val_avg_prob = 0.0
        # (small) reward for taking longer to crash
        time_elapsed = time.time() - start_timer
        fraction_of_allocated_time_passed_before_crashing = time_elapsed / ax_max_run_timer_seconds
        val_avg_prob = max(val_avg_prob, (0.1*fraction_of_allocated_time_passed_before_crashing))
        
        return val_avg_prob
    
    def train_experiment(parameters):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(train_experiment_thread, parameters)
            return future.result()
    
    best_parameters, best_values, experiment, model = optimize(
            parameters=[
                {"name": "decay_start", "type": "range", "bounds": [100, 40000], "value_type": "int", "log_scale": True},
                {"name": "lr_A", "type": "range", "bounds": [1e-5, 500e-5], "value_type": "float", "log_scale": True},
                {"name": "lr_B", "type": "range", "bounds": [100.0, 40000.0], "value_type": "float", "log_scale": True},
                {"name": "epochs_between_updates", "type": "range", "bounds": [1, 6], "value_type": "int", "log_scale": False},
                {"name": "decoder_rnn_dim", "type": "range", "bounds": [768, 1280], "value_type": "int", "log_scale": True},
                {"name": "p_attention_dropout", "type": "range", "bounds": [0, 0.5], "value_type": "float", "log_scale": False},
                {"name": "p_decoder_dropout", "type": "range", "bounds": [0, 0.5], "value_type": "float", "log_scale": False},
                {"name": "attention_rnn_dim", "type": "range", "bounds": [1024, 4096], "value_type": "int", "log_scale": True},
                {"name": "attention_dim", "type": "range", "bounds": [64, 512], "value_type": "int", "log_scale": True},
                {"name": "attention_location_n_filters", "type": "range", "bounds": [16, 128], "value_type": "int", "log_scale": True},
                {"name": "speaker_embedding_dim", "type": "range", "bounds": [64, 2048], "value_type": "int", "log_scale": True},
            ],
            # Booth function
            total_trials=15,
            evaluation_function=train_experiment,
            minimize=False,
            objective_name='validation_average_max_attention_weight',
        )
    print("__best_values__\n", best_values, "\n\n")
    print("__best_parameters__\n", best_parameters, "\n\n")
    print("__experiment__\n", experiment, "\n\n")
    print("__model__\n", model, "\n\n")
    
    filepath = "ax_experiment"
    print("saving experiment to;\n", filepath)
    ax.save(experiment, filepath)
    
