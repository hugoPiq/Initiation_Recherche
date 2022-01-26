""" NOT A REAL TRAINING SCRIPT
Please check the README.md in located in this same folder
for an explanation of this script"""

import gym
import gym_gazebo2
import time
env = gym.make('MYROBOT-v0')

while True:
    time.sleep(2000)
    # take a random action
    observation, reward, done, info = env.step([0, 1, ])
    time.sleep(5000)
    env.reset()
