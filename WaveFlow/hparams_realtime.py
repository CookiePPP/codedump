# ----------------------------------- LIVE PARAMS UPDATE ----------------------------------- # 
param_interval = 10
show_live_params = False

#learning_rate_ConvTranspose1d = 20e-5
#learning_rate_WN = 10e-5
#learning_rate_Invertible1x1Conv = 20e-5


custom_lr = True # enable Custom Learning Rates

# Custom LR
decay_start = 90000 # wait till decay_start to start decaying learning rate
A_ = 0.0002000
B_ = 60000
C_ = 0.0000000

warmup_start = 0
warmup_end =  100
warmup_start_lr = A_ * 0.5

# Scheduled LR
scheduler_patience = 40
scheduler_cooldown = 3
override_scheduler_best = False # WARNING: Override LR scheduler internal best value.
override_scheduler_last_lr = False #[A_+C_]

best_model_margin = 1.50
validation_interval = 250

# ----------------------------------- LIVE PARAMS UPDATE ----------------------------------- #