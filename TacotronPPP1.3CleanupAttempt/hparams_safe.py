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
        #ignore_layers=["decoder.attention_layer.F.2.weight", "decoder.attention_layer.F.2.bias","decoder.attention_layer.F.0.linear_layer.weight","decoder.attention_layer.F.0.linear_layer.bias"],
        ignore_layers=["encoder.lstm.weight_ih_l0","encoder.lstm.weight_hh_l0","encoder.lstm.bias_ih_l0","encoder.lstm.bias_hh_l0","encoder.lstm.weight_ih_l0_reverse","encoder.lstm.weight_hh_l0_reverse","encoder.lstm.bias_ih_l0_reverse","encoder.lstm.bias_hh_l0_reverse","decoder.attention_rnn.weight_ih","decoder.attention_rnn.weight_hh","decoder.attention_rnn.bias_ih","decoder.attention_rnn.bias_hh","decoder.attention_layer.query_layer.linear_layer.weight","decoder.attention_layer.memory_layer.linear_layer.weight","decoder.decoder_rnn.weight_ih","decoder.linear_projection.linear_layer.weight","decoder.gate_layer.linear_layer.weight"],
        
        ################################
        # Data Parameters              #
        ################################
        load_mel_from_disk=True,
        training_files='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/mel_train_taca2_merged.txt',
        validation_files='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/mel_validation_taca2_merged.txt',
        text_cleaners=['english_cleaners'],
        
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
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,
        
        # Gate
        gate_threshold=0.5,
        mask_gate_loss=False, # False = Vanilla Nvidia Tacotron2
        # masking the gate after the end of the clip will make the model never see the gate loss after the end of the clip. # TODO, explain this better # TODO, figure out why this is useful. # TODO, figure out why I added this
        # false would punish the model for trying to end the clip before it's ready, but barely punish the model for just forgetting to end the clip.
        # True will also help with badly trimmed audio.
        gate_positive_weight=10, # how much more valuable 1 positive frame is to 1 zero frame. 80 Frames per seconds, therefore values around 20 are fine.
        
        # Synthesis/Inference Related
        max_decoder_steps=3000,
        low_vram_inference=False, # doesn't save alignment and gate information, frees up some vram, especially for large input sequences.
        
        # Teacher-forcing Config
        p_teacher_forcing=1.00,    # 1.00 baseline
        teacher_force_till=20,     # int, number of starting frames with teacher_forcing at 100%, helps with clips that have challenging starting conditions i.e breathing before the text begins.
        val_p_teacher_forcing=0.80,
        val_teacher_force_till=20,
        
        # (Encoder) Encoder parameters
        encoder_speaker_embed_dim=256, # speaker_embedding before encoder
        encoder_concat_speaker_embed='inside', # concat before encoder convs, or just before the LSTM inside decode. Options 'before','inside'
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=768, #  = symbols_embedding_dim + encoder_speaker_embed_dim
        
        # (Decoder) Decoder parameters
        start_token = "",#"☺"
        stop_token = "",#"␤"
        hide_startstop_tokens=False, # remove first/last encoder output, *should* remove start and stop tokens from the decocer assuming the tokens are used.
        n_frames_per_step=1,    # currently only 1 is supported
        context_frames=1,   # TODO TODO TODO TODO TODO
        
        # (Decoder) Prenet
        prenet_dim=256,         # 256 baseline
        prenet_layers=2,        # 2 baseline
        prenet_batchnorm=False,  # False baseline
        p_prenet_dropout=0.5,   # 0.5 baseline
        
        # (Decoder) AttentionRNN
        attention_rnn_dim=1280, # 1024 baseline
        AttRNN_extra_decoder_input=True,# False baselinee
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
        torchMoji_attDim=2304,# pretrained model uses 2304
        torchMoji_linear=False,# load/save text infer linear layer.
        torchMoji_training=False,# switch GST to torchMoji mode
        
        # (GST) Drop Style Tokens
        p_drop_tokens=0.4, # Nudge the decoder to infer style without GST's input
        drop_tokens_mode='speaker_embedding',#Options: ('zeros','halfs','embedding','speaker_embedding') # Replaces style_tokens with either a scaler or an embedding, or a speaker_dependant embedding
        
        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=0.1e-5,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=56, # 32*3 = 0.377 val loss, # 2 = 0.71 val loss
        val_batch_size=56, # for more precise comparisons between models, constant batch_size is useful
        mask_padding=True,  # set model's padded outputs to padded values
        
        # DFR (Drop Frame Rate)
        global_mean_npy='global_mean.npy',
        drop_frame_rate=0.25,
        
        ##################################
        # MMI options                    #
        ##################################
        use_mmi=False,#depreciated
        use_gaf=True,#depreciated
        max_gaf=0.01,#depreciated
    )

    if hparams_string:
        tf.compat.v1.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.compat.v1.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
