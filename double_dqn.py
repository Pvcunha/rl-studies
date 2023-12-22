# paper reference: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import gymnasium as gym

from collections import deque
import random 

from itertools import count

device = "cuda" if torch.cuda.is_available() else "cpu" 
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005

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

        return torch.Tensor(sample_state).to(device), torch.tensor(sample_action, dtype=torch.int64).to(device), torch.Tensor(sample_reward).to(device), torch.Tensor(sample_nextstate).to(device), torch.tensor(sample_done, dtype=torch.bool).to(device)

    def __len__(self):
        return len(self.stateBuffer) # tanto faz a deque todas vao estar populadas da mesma forma

class Agent:
    def __init__(self, state_size, action_size, batch_size=64, seed=42, buffer_size=10000):
        
        self.state_size = state_size 
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.gamma = 0.99
        self.eps = 0.2
        self.lr = 1e-4

        self.policy_net = Net(state_size, action_size).to(device)
        self.target_net = Net(state_size, action_size).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.steps_done = 0
        
        self.buffer = ReplayBuffer(self.buffer_size)

    
    def choose_action(self, env, state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        if np.random.random() > eps_threshold:
            state_as_tensor = torch.tensor([state], dtype=torch.float32).to(device)
            actions = self.policy_net.forward(state_as_tensor)
            action = torch.argmax(actions).item()
        else:
            action = env.action_space.sample()

        return action

   
    def learn(self):

        s, action_batch, r, n_s, d = self.buffer.sample(self.batch_size)
        
        next_q = self.policy_net.forward(n_s).max(1)
        next_q = next_q.values.unsqueeze(1)

        next_q[d] = 0 # max stonks

        y = r.unsqueeze(1) + self.gamma*next_q 
        
        q_values = self.target_net(s)
        # breakpoint()
        q_values = q_values.gather(1, action_batch.unsqueeze(1))

        loss = F.smooth_l1_loss(q_values, y)
        self.optimizer.zero_grad()
        loss.backward()
        
        self.optimizer.step()

        return loss
    

env = gym.make("CartPole-v1", render_mode="human")

action_size = env.action_space.n
state, info = env.reset()
state_size = len(state)

agent = Agent(state_size, action_size)

for _ in range(1000):
    state, info = env.reset()
    done = False
    
    for t in count():
        # choose action
        action = agent.choose_action(env, state)

        _state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.buffer.push(state, action, reward, _state, done)
        if len(agent.buffer) >= agent.batch_size:
            loss = agent.learn()
            #print(loss.detach().cpu().item())
        
        policy_net_state_dict = agent.policy_net.state_dict()
        target_net_state_dict = agent.target_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + policy_net_state_dict[key]*(1-TAU)

        agent.target_net.load_state_dict(target_net_state_dict)

        state = _state
        
        if done:
            print(t+1)
            break

    # action = env.action_space.sample()

    # if terminated or truncated:
    #     observation, info = env.reset()


env.close()
