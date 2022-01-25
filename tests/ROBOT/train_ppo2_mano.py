"""
Script for testing that ppo2_mlp experiment is working
properly by taking hyperparameters from defaults
"""

import os
import sys
import time
import gym
import gym_gazebo2
import tensorflow as tf
import multiprocessing

from importlib import import_module
from baselines import bench, logger
from baselines.ppo2.ppo2 import learn
from baselines.ppo2 import model as ppo
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.policies import build_policy

ncpu = multiprocessing.cpu_count()

if sys.platform == 'darwin':
    ncpu //= 2

config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=ncpu,
                        inter_op_parallelism_threads=ncpu,
                        log_device_placement=False)

config.gpu_options.allow_growth = True

# Create environment
env_name = 'MYROBOT-v0'
env = DummyVecEnv([gym.make(env_name)])

# Hyperparamaters for ppo2 model
nminibatches = 4
nsteps = 1024
nbatch_act = 2  # num envs
nenvs = 1
ent_coef = 0.0
vf_coef = 1
nbatch = nenvs * nsteps
max_grad_norm = 0.5
nbatch_train = nbatch // nminibatches
ob_space = env.observation_space
ac_space = env.action_space
lr = 3e-4
network = "mlp"
total_timesteps = 1e8
gamma = 0.99
lam = 0.95
log_interval = 10
noptepochs = 4
cliprange = 0.2
num_layers = 2,
num_hidden = 64,
layer_norm = False
save_interval = 10
# Learn
learn(network=network, env=env, total_timesteps=total_timesteps, eval_env=None, seed=None, nsteps=nsteps, ent_coef=ent_coef, lr=lr,
      vf_coef=vf_coef,  max_grad_norm=max_grad_norm, gamma=gamma, lam=lam,
      log_interval=log_interval, nminibatches=nminibatches, noptepochs=noptepochs, cliprange=cliprange,
      save_interval=save_interval, load_path=None, model_fn=None, update_fn=None, init_fn=None, mpi_rank_weight=1, comm=None)

env.dummy().gg2().close()
os.kill(os.getpid(), 9)
