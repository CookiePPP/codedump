# ----------------------------------- LIVE PARAMS UPDATE ----------------------------------- # 
iteration = iteration # reminder that iteration var exists within this scope

param_interval = 5
show_live_params = False

#learning_rate_ConvTranspose1d = 20e-5
#learning_rate_WN = 10e-5
#learning_rate_Invertible1x1Conv = 20e-5

LossExplosionThreshold = 1e9

custom_lr = True # use Live Custom Learning Rate instead of Scheduler.

# Custom LR
decay_start = 600000 # wait till decay_start to start decaying learning rate
A_ = 0.0015000
B_ = 90000
C_ = 0.0000000

warmup_start = 0
warmup_end   = 1000
warmup_start_lr = 0.0002000

best_model_margin = 1.50 # training loss margin
validation_interval = 250#250 if iteration < 20000 else 1000

# Scheduled LR
patience_iterations = 10000 # number of iterations without improvement to decrease LR
scheduler_patience = int(patience_iterations/validation_interval)
scheduler_cooldown = 3
override_scheduler_best = False # WARNING: Override LR scheduler internal best value.
override_scheduler_last_lr = False #[A_+C_]

# ----------------------------------- LIVE PARAMS UPDATE ----------------------------------- #
