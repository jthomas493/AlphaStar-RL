from stable_baselines3 import PPO
import os
from sc2env import Sc2Env
import time
from wandb.integration.sb3 import WandbCallback
import wandb


# model_name = f"{int(time.time())}"

models_dir = "models/"
logdir = "logs/"
#
#
# conf_dict = {"Model": "v19",
#              "Machine": "Main",
#              "policy":"MlpPolicy",
#              "model_save_name": model_name}


# run = wandb.init(
# 	project="starcraft2",
# 	entity="jthomas493",
# 	config=conf_dict,
# 	sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
# 	save_code=True,  # optional
# 	)


# if not os.path.exists(models_dir):
# 	os.makedirs(models_dir)
#
# if not os.path.exists(logdir):
# 	os.makedirs(logdir)

LOAD_MODEL = 'Abyssal.zip'

env = Sc2Env()

# load the model:
model = PPO.load(LOAD_MODEL, env=env, tensorboard_log='logs/no_reward_run' )

TIMESTEPS = 10000000
iters = 0
while True:
	print("On iteration: ", iters)
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name='run',
				callback= model.save('reward_PPO'))
	model.save('Abyssal_PPO')
