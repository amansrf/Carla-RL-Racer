import torch as th

# imports for file path handling
import os
import sys
from pathlib import Path

sys.path.append(Path(os.getcwd()).parent.as_posix())

misc_params = {
  "env_name": 'roar-e2e-ppo-v0',
  "run_fps": 32,  # TODO Link to the environment RUN_FPS
  "model_directory": Path("./output/SAC_test3"),
  "run_name": "SAC_test3",
  "total_timesteps": int(1e6),#1e6
}

spawn_params = {
  "num_spawn_pts": 12,  # Last one is 12 but tianlun did not fixed 12
  "init_spawn_pt": 0,
  "dynamic_spawn": True,  # True if start at different spawn locations on reset
  "spawn_position" : {
                      0 : (2542.800048828125,111.44281768798828,4069.599853515625),
                      1 : (1350.7999267578125,61.94282531738281,4260.10009765625),
                      2 : (1328.2999267578125,68.84282684326172,3553.099853515625),
                      3 : (2421.800048828125,115.44281768798828,3741.599853515625),
                      4 : (3592.799560546875,192.74282836914062,3660.099853515625),
                      5 : (3043.300048828125,345.4428405761719,2867.099853515625),
                      6 : (3527.799560546875,417.2428283691406,2609.099853515625),
                      7 : (4836.2998046875,505.0428161621094,2955.599853515625),
                      8 : (5272.2998046875,500.2428283691406,3083.599853515625),
                      9 : (5614.7998046875,422.7428283691406,4085.599853515625),
                      10: (3695.799560546875,145.84280395507812,4926.10009765625),
                      11: (3090.300048828125,116.142822265625,4852.60009765625),
                      12: (2596.300048828125,103.54281616210938,4359.10009765625)
                    },


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
  "custom_list": [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # List of custom spawn pts                                                                                                                                                                                                                      
  "spawn_pt_iterator": 1,  # DO NOT TOUCH THIS! Used Internally!
  # "spawn_int_map": [39, 91, 140, 224, 312, 442, 556, 730, 782, 898, 1142, 1283, 3],
  "spawn_int_map": [70, 317, 467, 727, 993, 1269, 1512, 1906, 2024, 2298, 2923, 3179, 4],
}

wandb_saves = {
  "gradient_save_freq": 256 * misc_params["run_fps"],
  "model_save_freq": 256 * misc_params["run_fps"],
}

PPO_params = dict(
  learning_rate = 0.0005,  # be smaller 2.5e-4
  n_steps = 256 * misc_params["run_fps"],#1024
  batch_size=256,  # mini_batch_size = 256?
  # n_epochs=10,
  gamma=1,  # rec range .9 - .99 0.999997
  ent_coef=.00,  # rec range .0 - .01
  # gae_lambda=0.95,
  # clip_range_vf=None,
  # vf_coef=0.5,
  # max_grad_norm=0.5,
  # use_sde=True,
  # sde_sample_freq=misc_params["run_fps"]/2,
  # target_kl=None,
  # tensorboard_log=(Path(misc_params["model_directory"]) / "tensorboard").as_posix(),
  # create_eval_env=False,
  # policy_kwargs=None,
  verbose=1,
  seed=1,
  device=th.device('cuda' if th.cuda.is_available() else 'cpu'),
  # _init_setup_model=True,
)

SAC_params = dict(
  learning_rate = 1e-5,
  batch_size=256,
  # n_steps = misc_params["run_fps"],#1024
  ent_coef="auto",
  target_entropy="auto",
  gamma=0.99,
  use_sde=True,
  sde_sample_freq=5*misc_params["run_fps"],
  buffer_size=256_000, #default 1_000_000,
  verbose=1,
  seed=1,
  device=th.device('cuda' if th.cuda.is_available() else 'cpu'),
)
