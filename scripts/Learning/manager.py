import random
from training import MyEnvironment, MyNeuralNetwork
import torch
import torch.nn as nn
import math

# Instantiate the environments and the neural network
env = MyEnvironment()
policy_net = MyNeuralNetwork()

# Set the parameters for the learning algorithm
num_episodes = 1000
batch_size = 32
gamma = 0.99
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 0.995

# Set up the optimizer and loss function
optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)
mse_loss = nn.MSELoss()

# Start the learning loop
for i_episode in range(num_episodes):
    # Reset the environment and get the initial state
    state = env.reset()
    done = False
    
    # Initialize the episode's total reward and time steps
    episode_reward = 0
    t = 0
    
    while not done:
        # Choose the action based on epsilon-greedy policy
        epsilon = epsilon_final + (epsilon_start - epsilon_final) * \
            math.exp(-1.0 * t / epsilon_decay)
        if random.random() < epsilon:
            action = env.sample_action()
        else:
            action = policy_net.choose_action(state)
        
        # Take the action and observe the next state and reward
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        
        # Store the experience in the replay buffer
        policy_net.replay_buffer.add(state, action, reward, next_state, done)
        
        # Sample a minibatch of experiences and train the neural network
        if len(policy_net.replay_buffer) >= batch_size:
            batch = policy_net.replay_buffer.sample(batch_size)
            loss = 0.0
            
            for s, a, r, ns, d in batch:
                target = r + gamma * policy_net.choose_action(ns)
                if not d:
                    target += gamma * policy_net.choose_action(ns)
                q_value = policy_net.forward(s)[a]
                loss += mse_loss(target, q_value)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Update the state and time steps
        state = next_state
        t += 1
        
    # Print the episode's total reward
    print("Episode {}: Total Reward = {}".format(i_episode, episode_reward))
