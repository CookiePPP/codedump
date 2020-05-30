# ----------------------------------- LIVE PARAMS UPDATE ----------------------------------- # 
param_interval = 5
show_live_params = False

#learning_rate_ConvTranspose1d = 20e-5
#learning_rate_WN = 10e-5
#learning_rate_Invertible1x1Conv = 20e-5

LossExplosionThreshold = 1e2

custom_lr = True # enable Custom Learning Rates

# Custom LR
decay_start = 300000 # wait till decay_start to start decaying learning rate
A_ = 0.0001000
B_ = 90000
C_ = 0.0000000

warmup_start = 0
warmup_end   = 2000
warmup_start_lr = 0.0001000

best_model_margin = 1.50
validation_interval = 100

# Scheduled LR
patience_iterations = 10000 # number of iterations without improvement to decrease LR
scheduler_patience = int(patience_iterations/validation_interval)
scheduler_cooldown = 3
override_scheduler_best = False # WARNING: Override LR scheduler internal best value.
override_scheduler_last_lr = False #[A_+C_]

# ----------------------------------- LIVE PARAMS UPDATE ----------------------------------- #
