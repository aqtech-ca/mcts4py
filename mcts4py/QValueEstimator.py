import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class StateValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StateValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)

class StateActionValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StateActionValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state_action):
        x = torch.relu(self.fc1(state_action))
        return self.fc2(x)

class QValueEstimator:
    def __init__(self, state_dim, action_dim, hidden_dim, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.state_value_net = StateValueNetwork(state_dim, hidden_dim, 1)
        self.q_net = StateActionValueNetwork(state_dim + action_dim, hidden_dim, 1)
        self.optimizer_state_value = optim.Adam(self.state_value_net.parameters(), lr=0.001)
        self.optimizer_q_value = optim.Adam(self.q_net.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.q_val_dict = {}
        self.state_val_dict = {}
        self.alpha = alpha
    
    def predict_state_value(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        return self.state_value_net(state)
    
    def predict_q_value(self, state, action):
        state_action = torch.tensor(np.concatenate((state, action)), dtype=torch.float32)
        return self.q_net(state_action)
    
    def get_max_q_value(self, state, possible_actions):
        max_q_value = -np.inf
        for a in possible_actions:
            state_action = torch.tensor(np.concatenate((state, a)), dtype=torch.float32)
            candidate_q_value = self.q_net(state_action)
            if candidate_q_value > max_q_value:
                max_q_value = candidate_q_value

        return max_q_value
    
    def update_state_value(self, state, possible_actions):
        soft_q_value = 
        self.state_val_dict[repr(state)] = max_q_value
        return max_q_value
    
    def update_q_value(self, state, action, reward, reward_to_go):
        self.q_val_dict[repr(state, action)] = reward + reward_to_go
    
    # def forward_bellman_update(self, state, action, reward, next_state, done, discount_factor):
    #     next_state_value = self.predict_state_value(next_state).item()
    #     target = reward + discount_factor * next_state_value * (1 - done)
    #     self.update_state_action_value(state, action, target)

# self.optimizer_state_value.zero_grad()
# loss = self.loss_fn(max_q_value, max_q_value)
# loss.backward()
# self.optimizer_state_value.step()

# def update_q_value(self, state, action, target):
#     state_action = torch.tensor(np.concatenate((state, action)), dtype=torch.float32)
#     target = torch.tensor(target, dtype=torch.float32)
#     self.optimizer_q_value.zero_grad()
#     predicted = self.q_net(state_action)
#     loss = self.loss_fn(predicted, target)
#     loss.backward()
#     self.optimizer_q_value.step()

# def softmax_policy_update(self, state, action_values, temperature=1.0):
#     action_probs = nn.functional.softmax(action_values / temperature, dim=0)
#     chosen_action = torch.multinomial(action_probs, num_samples=1).item()
#     chosen_action_prob = action_probs[chosen_action]
#     target = self.predict_state_value(state).detach().item()  # Detach to prevent gradients from flowing
#     error = target - chosen_action_prob
#     self.update_state_action_value(state, chosen_action, error)