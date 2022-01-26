""" NOT A REAL TRAINING SCRIPT
Please check the README.md in located in this same folder
for an explanation of this script"""

import gym
import gym_gazebo2
import time
env = gym.make('MYROBOT-v0')
print(env._observation_msg)
while True:
    observation, reward, done, info = env.step([0, 1])
