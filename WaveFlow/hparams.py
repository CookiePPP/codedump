import tensorflow as tf


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Quality of Life Parameters   #
        ################################
        tqdm=True, # enable Progress Bars
        notebook=False, # Use notebook specific features (Progress Bars, Live Graphs, Embedded Audio, etc)
        
        ################################
        # Experiment Parameters        #
        ################################
        epochs=1000,
        iters_per_checkpoint=5000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        fp16_opt_level='01', # '00' = fp32, '01' = fp16 with fp32 for anything that needs precision (recommended for this model), '02' = fp16 for almost everything, '03' = Full fp16 with no scaling - very unstable.
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://127.0.0.1:54321",
        cudnn_enabled=True,
        cudnn_benchmark=True,
        
        ################################
        # Dataloader Parameters        #
        ################################
        training_files='windows_filelists/map_0_GT.txt',
        #training_files='filelists/map_0_GT.txt',
        segment_length=24000,
        load_mel_from_disk=False,
        cache_spectrograms=False, # saves processed spectrograms to disk to load later (thereby saving CPU time). Audio will be aligned along the hop_length if using cache, so be aware of a potential quality differences.
        preempthasis=0.00,
        n_dataloader_workers=2,
        
        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=48000,
        filter_length=2400,
        hop_length=600,
        win_length=2400,
        n_mel_channels=160,
        mel_fmin=0.0,
        mel_fmax=16000.0,
        
        ################################
        # Validation Parameters        #
        ################################
        validation_files='windows_filelists/map_0_GT_Val.txt',
        #validation_files='filelists/map_0_GT_Val.txt',
        
        ################################
        # Test Parameters              #
        ################################
        test_files='filelists/map_0_GT_Val.txt',
        validation_windows=[1200, 2400], # windows lengths to use for spectrogram MSE.
        
        ################################
        # Model Parameters             #
        ################################

        # Core Parameters
        core_source = "yoyo", # options "NVIDIA","yoyo"
        core_type = "WaveGlow", # options "WaveGlow","WaveFlow"
        n_early_every = 12,
        n_early_size = 2,
        sigma = 0.07,
        
        # InvertableConv Parameters
        InvertableConv_source = "yoyo", # options "NVIDIA","yoyo","None"
        InvConv_memory_efficient = True,
        
        # AffineCouplingBlock Parameters
        AffineCouplingBlock_source = "yoyo", # options "NVIDIA","yoyo"
        ACB_memory_efficient = True,
        
        # WN Parameters
        WN_source = "NVIDIA",# options "NVIDIA","yoyo"
        n_flows = 12,
        n_group = 8,
        n_layers = 9,
        n_channels = 256,
        kernel_width = 3,
        kernel_height = 3,#WaveFlow only
        max_speakers = 512,
        speaker_embed_dim = 96,
        rezero = False, # experimental
        
        # Spectrogram Upsampler
        upsampler_source = "L0SG", # options "NVIDIA","yoyo","CookiePPP","L0SG"
        upsample_factors = [24,25], # 600x upsample scale
        upsample_groups = 1, # Applicable for conv based upscalers only.
        
        # Audio & Spectrogram Squeezer
        squeezer_source = "yoyo", # options "NVIDIA","yoyo","L0SG","PaddlePaddle"
        
        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        LRScheduler="ReduceLROnPlateau",
        optimizer='Adam',
        optimizer_fused=False, # take optimizer from Nvidia/Apex, uses **slightly** more VRAM but faster.
        learning_rate=0.1e-5,
        weight_decay=1e-6,
        b_grad_clip=False, # should gradients be clipped (slightly higher time/iter loss)
        grad_clip_thresh=100.0, # what to clip gradients too
        batch_size=8,
    )

    if hparams_string:
        tf.compat.v1.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.compat.v1.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
