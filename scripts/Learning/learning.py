import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.q_network = QNetwork(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

    def select_action(self, state, epsilon):
        if torch.rand(1) > epsilon:
            # Use Q-network to select discrete or continuous action
            q_values = self.q_network(state)
            discrete_action = torch.argmax(q_values[:4])
            continuous_action = q_values[4:]
            action = torch.cat([discrete_action, continuous_action], dim=0)
        else:
            # Randomly select action
            discrete_action = torch.randint(4, size=(1,))
            continuous_action = torch.randn(3)
            action = torch.cat([discrete_action, continuous_action], dim=0)
        return action

    def train_step(self, states, actions, rewards, next_states, terminals, gamma):
        # Compute Q targets
        q_targets = rewards + gamma * torch.max(self.q_network(next_states), dim=1)[0] * (1 - terminals)

        # Compute Q values for selected actions
        q_values = self.q_network(states)
        discrete_actions = actions[:, 0].long()
        continuous_actions = actions[:, 1:]
        selected_q_values = q_values.gather(1, discrete_actions.unsqueeze(1)).squeeze()
        continuous_indices = torch.arange(4, 7).unsqueeze(0).repeat(len(states), 1).to(continuous_actions.device)
        selected_q_values[continuous_indices] = continuous_actions.sum(dim=1)

        # Compute loss and update Q-network
        loss = nn.SmoothL1Loss()(selected_q_values, q_targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Define parameters
input_dim = 15  # size of observation space
hidden_dim = 64
output_dim = 7  # size of action space
epsilon = 0.1
gamma = 0.99
batch_size = 32
num_episodes = 1000
max_steps_per_episode = 100

# Create agent
agent = Agent(input_dim, hidden_dim, output_dim)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    for step in range(max_steps_per_episode):
        # Select action
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = agent.select_action(state_tensor, epsilon)
        discrete_action = int(action[0])
        continuous_action = action[1:]

        # Take action and observe reward and next state
        if discrete_action == 0:
            # Move forward
            reward, next_state, done = env.move_forward()
        elif discrete_action == 1:
            # Turn left
            reward, next_state, done = env.turn
