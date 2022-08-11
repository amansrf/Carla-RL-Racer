import torch as th

# imports for file path handling
import os
import sys
from pathlib import Path

sys.path.append(Path(os.getcwd()).parent.as_posix())

misc_params = {
  "env_name": 'roar-e2e-ppo-v0',
  "run_fps": 32,  # TODO Link to the environment RUN_FPS
  "model_directory": Path("./output/PPOe2e_Major_FC_Run_9 -- 32fpss_fullthrottle_largeroverlapdetection"),
  "run_name": "Major FC Run 9",
  "total_timesteps": int(1e6),
}

spawn_params = {
  "num_spawn_pts": 13,  # Last one is 12s
  "init_spawn_pt": 1,
  "dynamic_spawn": True,  # True if start at different spawn locations on reset

  # Spawn Guide:
  # 1 = Roundabout
  # 5 = Sharpest Turn
  # 8 = Leap of Faith (Height of Leap is roughly 360)
  # 12 = "Race Start"

  # Dynamic Type Choice:
  #   1. "uniform random" - Choose from uniform random distribution in range(init_spawn_point:num_spawn_pts)
  #   2. "linear forward" - After reset spawn point increments by one. Loops back to init after num_spawn_pts reached
  #   3. "linear backward" - After reset decrement spawn point by one. Loops back to num_spawn_pts after init reached
  #   4. "custom spawn pts" - Provide a custom list of spawn points.
  "dynamic_type": "uniform random",
  "custom_list": [4, 9, 0, 12, 7],  # List of custom spawn pts

  "spawn_pt_iterator": 0,  # DO NOT TOUCH THIS! Used Internally!
  "spawn_int_map": [91, 0, 140, 224, 312, 442, 556, 730, 782, 898, 1142, 1283, 39],
}

wandb_saves = {
  "gradient_save_freq": 512 * misc_params["run_fps"] * 10,
  "model_save_freq": 50 * misc_params["run_fps"] * 10,
}

# def lr_scheduler(init_lr : float):
#   counter = 1
#   #decaying_rate = 0.85
#   update_step = 20
#   #current_lr = init_lr
#   def update_lr(uesless_arg):
#     # global counter
#     # global decaying_rate
#     # global update_step
#     # global current_lr
#     if counter % update_step == 0:
#        current_lr *= decaying_rate
#     return current_lr
#   return update_lr    


PPO_params = dict(
  learning_rate = 0.00001,  # be smaller 2.5e-4
  # lr_deacying_rate = 0.85, 
  # lr_decaying_step = 1000000, 
  # learning_rate = lr_scheduler(0.00001), 
  n_steps = 1024 * misc_params["run_fps"],
  batch_size=64,  # mini_batch_size = 256?
  # n_epochs=10,
  gamma=0.99,  # rec range .9 - .99
  ent_coef=.00,  # rec range .0 - .01
  # gae_lambda=0.95,
  # clip_range_vf=None,
  # vf_coef=0.5,
  # max_grad_norm=0.5,
  # use_sde=True,
  # sde_sample_freq=5,
  # target_kl=None,
  # tensorboard_log=(Path(misc_params["model_directory"]) / "tensorboard").as_posix(),
  # create_eval_env=False,
  # policy_kwargs=None,
  verbose=1,
  seed=1,
  device=th.device('cuda:0' if th.cuda.is_available() else 'cpu'),
  # _init_setup_model=True,
)
