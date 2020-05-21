import tensorflow as tf
from text.symbols import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=1000,
        iters_per_checkpoint=1000,
        iters_per_validation=1000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://127.0.0.1:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=[None],
        
        ################################
        # Data Parameters              #
        ################################
        load_alignments=True,# required for PAG (Pre-alignment Guided Attention), otherwise disable
        load_mel_from_disk=True,# otherwise, load mel from audio file
        cache_mels=True, # save mel-spec alongside audiofiles to save CPU processing time.
        speakerlist='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/speaker_ids.txt',
        training_files='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/mel_train_taca2_merged.txt',
        validation_files='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/mel_validation_taca2_merged.txt',
        text_cleaners=['basic_cleaners'],
        
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
        # Fundamental frequency (f0)   #
        ################################
        f0_prenet=True, # allow f0 input
        f0_prediction=False, # predict f0 of next timestep (note, both prenet and prediction are needed for autoregressive f0)
        f0_min=80,
        f0_max=880,
        harm_thresh=0.25,
        
        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,
        
        # Gate
        gate_positive_weight=10, # how much more valuable 1 positive frame is to 1 zero frame. 80 Frames per seconds, therefore values around 20 are fine.
        
        # Synthesis/Inference Related
        gate_threshold=0.5,
        gate_delay=10,
        max_decoder_steps=3000,
        low_vram_inference=False, # doesn't save alignment and gate information, frees up some vram, especially for large input sequences.
        
        # Teacher-forcing Config
        p_teacher_forcing=1.00,    # 1.00 baseline
        teacher_force_till=20,     # int, number of starting frames with teacher_forcing at 100%, helps with clips that have challenging starting conditions i.e breathing before the text begins.
        val_p_teacher_forcing=0.80,
        val_teacher_force_till=20,
        
        # (Encoder) Encoder parameters
        encoder_speaker_embed_dim=256,
        encoder_concat_speaker_embed='before_lstm', # concat before encoder convs, or just before the LSTM inside decode. Options 'before_conv','before_lstm'
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_conv_hidden_dim=512,
        encoder_LSTM_dim=768,
        
        # (Decoder) Decoder parameters
        start_token = "",#"☺"
        stop_token = "",#"␤"
        hide_startstop_tokens=False, # trim first/last encoder output before feeding to attention.
        n_frames_per_step=1,    # currently only 1 is supported
        context_frames=1,   # TODO TODO TODO TODO TODO
        
        # (Decoder) Prenet
        prenet_dim=256,         # 256 baseline
        prenet_layers=2,        # 2 baseline
        prenet_batchnorm=False,  # False baseline
        p_prenet_dropout=0.5,   # 0.5 baseline
        
        # (Decoder) AttentionRNN
        attention_rnn_dim=1280, # 1024 baseline
        AttRNN_extra_decoder_input=True,# False baseline
        AttRNN_hidden_dropout_type='zoneout',# options ('dropout','zoneout')
        p_AttRNN_hidden_dropout=0.10,     # 0.1 baseline
        p_AttRNN_cell_dropout=0.00,       # 0.0 baseline
        
        # (Decoder) AttentionRNN Speaker embedding
        n_speakers=512,
        speaker_embedding_dim=256, # speaker embedding size # 128 baseline
        
        # (Decoder) DecoderRNN
        decoder_rnn_dim=1024,   # 1024 baseline
        extra_projection=False, # another linear between decoder_rnn and the linear projection layer (hopefully helps with high sampling rates and hopefully doesn't help decoder_rnn overfit)
        DecRNN_hidden_dropout_type='zoneout',# options ('dropout','zoneout')
        p_DecRNN_hidden_dropout=0.1,     # 0.1 baseline
        p_DecRNN_cell_dropout=0.00,       # 0.0 baseline
        
        # (Decoder) Attention parameters
        attention_type=0,
        # 0 -> Location-Based Attention (Vanilla Tacotron2)
        # 1 -> GMMAttention (Multiheaded Long-form Synthesis)
        attention_dim=128,      # 128 Layer baseline
        
        # (Decoder) Attention Type 0 Parameters
        attention_location_n_filters=32,   # 32 baseline
        attention_location_kernel_size=31, # 31 baseline
        
        # (Decoder) Attention Type 1 Parameters
        num_att_mixtures=1,# 5 baseline
        attention_layers=1,# 1 baseline
        delta_offset=0,    # 0 baseline, values around 0.005 will push the model forwards. Since we're using the sigmoid function caution is suggested.
        delta_min_limit=0, # 0 baseline, values around 0.010 will force the model to move forward, in this example, the model cannot spend more than 100 steps on the same encoder output.
        lin_bias=False, # I need to figure out what that layer is called.
        initial_gain='relu', # initial weight distribution 'tanh','relu','sigmoid','linear'
        normalize_attention_input=True, # False baseline
        normalize_AttRNN_output=False,  # True baseline
        
        # (Postnet) Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
        
        # (GST) Reference encoder
        with_gst=True,
        ref_enc_pack_padded_seq=True, # TODO
        ref_enc_filters=[32, 32, 64, 64, 128, 128],
        ref_enc_size=[3, 3],
        ref_enc_strides=[2, 2],
        ref_enc_pad=[1, 1],
        ref_enc_gru_size=128,
        
        # (GST) Multi-headed Attention Layer
        gstAtt_dim=128,
        num_heads=8,
        
        # (GST) Style Token Layer
        token_num=5, # acts as the information bottleneck.
        token_activation_func='tanh', # default 'softmax', options 'softmax','sigmoid','tanh','absolute'
        token_embedding_size=256, # token embedding size
        
        # (GST) TorchMoji
        torchMoji_attDim=2304,# published model uses 2304
        torchMoji_linear=True,# load/save text infer linear layer.
        torchMoji_training=True,# switch GST to torchMoji mode
        
        # (GST) Drop Style Tokens
        p_drop_tokens=0.0, # Nudge the decoder to infer style without GST's input
        drop_tokens_mode='zeros',#Options: ('zeros','halfs','embedding','speaker_embedding','emotion_embedding') # Replaces style_tokens with either a scaler or an embedding, or speaker embeddings
        
        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        loss_func = 'MSELoss', # options 'MSELoss','SmoothL1Loss'
        learning_rate=0.1e-5,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=32,
        val_batch_size=32, # for more precise comparisons between models, constant batch_size is required # TODO, calculate mean loss without paddings
        truncated_length=640, # max mel length till truncation.
        mask_padding=True,
        
        # (DFR) Drop Frame Rate
        global_mean_npy='global_mean.npy',
        drop_frame_rate=0.25,
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
