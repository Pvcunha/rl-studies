import numpy as np
import gymnasium as gym

env = gym.make('FrozenLake-v1', render_mode="ansi", is_slippery=False)

os = env.observation_space
print(os)
print(env.action_space.n)

qtable = np.random.random((env.observation_space.n, env.action_space.n))

alpha = 1e-3
gamma = 0.99

eps = 1.0
for step in range(1000):
    
    curr_state, info = env.reset()
    done = False
    #print(steps)
    while not done:
        prob = np.random.rand()
        if prob < 0.85:
            action = np.argmax(qtable[curr_state])
        else:
            action = env.action_space.sample()
            

        new_state, reward, done, truncated, info = env.step(action)

        qtable[curr_state][action] = qtable[curr_state][action] + alpha*reward + gamma*(np.max(qtable[new_state]) - qtable[curr_state][action]) 
        
        if reward > 0:
            print('GANHOU PLAYBOY', step)
        curr_state = new_state
        env.render()


env.close()
    
