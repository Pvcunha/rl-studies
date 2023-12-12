import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import gymnasium as gym

from collections import deque
import random 

device = "cuda" if torch.cuda.is_available() else "cpu" 

class Net(nn.Module):
    def __init__(self, state_size, action_size):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(state_size, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        return self.layer3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.stateBuffer = deque([], capacity)
        self.actionBuffer = deque([], capacity)
        self.rewardBuffer = deque([], capacity)
        self.nextStateBuffer = deque([], capacity)
        self.doneBuffer = deque([], capacity)

    def push(self, state, action, reward, next_state, done):
        self.stateBuffer.append(state)
        self.actionBuffer.append(action)
        self.rewardBuffer.append(reward)
        self.nextStateBuffer.append(next_state)
        self.doneBuffer.append(done)


    def sample(self, batch_size):
        sample_state = random.sample(self.stateBuffer, batch_size)
        sample_action = random.sample(self.actionBuffer, batch_size)
        sample_reward = random.sample(self.rewardBuffer, batch_size)
        sample_nextstate = random.sample(self.nextStateBuffer, batch_size)
        sample_done = random.sample(self.doneBuffer, batch_size)

        return sample_state, sample_action, sample_reward, sample_nextstate, sample_done

    def __len__(self):
        return len(self.stateBuffer) # tanto faz a deque todas vao estar populadas da mesma forma

class Agent:
    def __init__(self, state_size, action_size, batch_size=64, seed=42):
        
        self.state_size = state_size 
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        
        self.net = Net(state_size, action_size).to(device)
        
        self.buffer = ReplayBuffer(10000)

    
    def choose_action(self, env, state):
        
        if np.random.random() > self.eps:
            state_as_tensor = torch.tensor([state], dtype=torch.float).to(self.qNet_local.device)
            actions = self.net.forward(state_as_tensor)
            action = torch.argmax(actions).item()
        else:
            action = env.action_space.sample()

        return action

   
    def learn(self):
       if len(self.buffer) < self.batch_size:
           return

       s, a, r, n_s, d = self.buffer.sample(self.batch_size)
       
       pass
    

env = gym.make("CartPole-v1", render_mode="human")

action_size = env.action_space.n

state, info = env.reset()
state_size = len(state)
done = False

agent = Agent(state_size, action_size)

for _ in range(1000):

    while not done:
        # choose action
        action = agent.choose_action()
        _state, reward, terminated, truncated, info = env.step(action)
        _done = terminated or truncated
        agent.buffer.push(state, action, reward, _state, _done)
        agent.learn()

    action = env.action_space.sample()

    if terminated or truncated:
        observation, info = env.reset()


env.close()
