import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("CartPole-v1", render_mode="human" )

curr_state, info = env.reset()

print(env.action_space)
done = False
while not done:

    action = env.action_space.sample()
    new_state, reward, done, truncated, info = env.step(action)
    env.render()

env.close()


