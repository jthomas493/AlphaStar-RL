# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 20:51:42 2022

@author: james
"""
# use ppo2 to learn and save the model when finished
from stable_baselines3 import PPO
import os
from sc2env import Sc2Env
import time


LOAD_MODEL = 'models/1647915989/2880000.zip'

env = Sc2Env()

# load the model:
model = PPO.load(LOAD_MODEL, env=env, verbose=1, tensorboard_log="log/")

TIMESTEPS = 1000000
iters = 0
while True:
	print("On iteration: ", iters)
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="abyssal_run", callback= model.save('Abyssal'))
	model.save("models/Abyssal_PPO")



