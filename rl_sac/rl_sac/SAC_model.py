import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, value_lr,device, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.lr = value_lr
        self.dev = device
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2_3 = nn.Linear(hidden_dim, hidden_dim//2)
        self.linear3 = nn.Linear(hidden_dim//2, 1)
        
        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)
        self.linear2_3.weight.data.uniform_(-init_w, init_w)
        self.linear2_3.bias.data.uniform_(-init_w, init_w)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
        self.to(self.dev)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    def forward(self, state):
        x = mish(self.linear1(state))
        x = mish(self.linear2(x))
        x = mish(self.linear2_3(x))
        x = self.linear3(x)
        return x
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size,actor_lr,DEVICE,init_w=3e-3, log_std_min=-10, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, hidden_size)
        self.fc_1.weight.data.uniform_(-init_w, init_w)
        self.fc_1.bias.data.uniform_(-init_w, init_w)

        self.fc_2 = nn.Linear(hidden_size, hidden_size)
        self.fc_2.weight.data.uniform_(-init_w, init_w)
        self.fc_2.bias.data.uniform_(-init_w, init_w)

        self.fc_3 = nn.Linear(hidden_size, hidden_size//2)
        self.fc_3.weight.data.uniform_(-init_w, init_w)
        self.fc_3.bias.data.uniform_(-init_w, init_w)

        self.fc_mu = nn.Linear(hidden_size//2, action_dim)
        self.fc_mu.weight.data.uniform_(-init_w, init_w)
        self.fc_mu.bias.data.uniform_(-init_w, init_w)

        self.fc_log_std = nn.Linear(hidden_size//2, action_dim)
        self.fc_log_std.weight.data.uniform_(-init_w, init_w)
        self.fc_log_std.bias.data.uniform_(-init_w, init_w)

        self.lr = actor_lr
        self.dev = DEVICE

        self.LOG_STD_MIN = log_std_min
        self.LOG_STD_MAX = log_std_max
        self.max_action = torch.FloatTensor([ACTION_W_MAX, ACTION_V_MAX]).to(self.dev)
        self.min_action = torch.FloatTensor([ACTION_W_MIN, ACTION_V_MIN]).to(self.dev)
        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias = (self.max_action + self.min_action) / 2.0
        
        self.to(self.dev)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = F.leaky_relu(self.fc_1(x))
        x = F.leaky_relu(self.fc_2(x))
        x = F.leaky_relu(self.fc_3(x))
        mu = self.fc_mu(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, state,epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        reparameter = Normal(mean, std)
        z = reparameter.rsample()
        y_t = torch.tanh(z)
        action = self.action_scale * y_t + self.action_bias

        # # # Enforcing Action Bound
        log_prob = reparameter.log_prob(z)
        log_prob = log_prob - torch.sum(torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon), dim=-1, keepdim=True)
        # action = torch.tanh(z)
        # log_prob = reparameter.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        # log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size,critic_lr,device, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2_3 = nn.Linear(hidden_size, hidden_size//2)
        self.linear3 = nn.Linear(hidden_size//2, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=critic_lr)
        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)
        self.linear2_3.weight.data.uniform_(-init_w, init_w)
        self.linear2_3.bias.data.uniform_(-init_w, init_w)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        self.device = device
        self.to(self.device)
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = mish(self.linear1(x))
        x = mish(self.linear2(x))
        x = mish(self.linear2_3(x))
        x = self.linear3(x)
        return x
def mish(x):
    '''
        Mish: A Self Regularized Non-Monotonic Neural Activation Function
        https://arxiv.org/abs/1908.08681v1
        implemented for PyTorch / FastAI by lessw2020
        https://github.com/lessw2020/mish
        param:
            x: output of a layer of a neural network
        return: mish activation function
    '''
    return torch.clamp(x*(torch.tanh(F.softplus(x))),max=6)