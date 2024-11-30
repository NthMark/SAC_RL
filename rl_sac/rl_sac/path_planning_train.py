import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist,Point
from nav_msgs.msg import OccupancyGrid
import numpy as np
from .display_waypoints import WaypointPublisher
from .grid_map import GridMap
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import os
import time
from collections import deque
import random
SAMPLE_THRESHOLD = 500
ACTION_DIMENSION = 4 
STATE_DIMENSION  = 3 # current_position->2, distance->1, 5x5 neighbors_status->25, 
EPISODE=5000
MAX_STEP_SIZE=2000
HIDDEN_SIZE =256
class ReplayBuffer:
    def __init__(self, buffer_limit, device):
        self.buffer = deque(maxlen=buffer_limit)
        self.device = device

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([0.0 if done else 1.0])

        s_batch = torch.tensor(np.array(s_lst), dtype=torch.float).to(self.device)
        a_batch = torch.tensor(np.array(a_lst), dtype=torch.float).to(self.device)
        r_batch = torch.tensor(np.array(r_lst), dtype=torch.float).to(self.device)
        s_prime_batch = torch.tensor(np.array(s_prime_lst), dtype=torch.float).to(self.device)
        done_batch = torch.tensor(np.array(done_mask_lst), dtype=torch.float).to(self.device)

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def size(self):
        return len(self.buffer)
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

        self.fc_out = nn.Linear(hidden_size//2, action_dim)
        self.fc_out.weight.data.uniform_(-init_w, init_w)
        self.fc_out.bias.data.uniform_(-init_w, init_w)

        self.lr = actor_lr
        self.dev = DEVICE
        
        self.to(self.dev)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = F.leaky_relu(self.fc_1(x))
        x = F.leaky_relu(self.fc_2(x))
        x = F.leaky_relu(self.fc_3(x))
        logits = self.fc_out(x)
        return F.softmax(logits, dim=-1) 

    def sample(self, state):
        action_probs = self.forward(state)
        distribution = Categorical(action_probs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
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
        if action.dim() == 1:
            action = action.unsqueeze(1)  # Ensure action has the same number of dimensions as state
        action = action.to(torch.long)
        action_one_hot = F.one_hot(action, num_classes=ACTION_DIMENSION).float()  
        action_one_hot = action_one_hot.squeeze(1) 
        x = torch.cat([state, action_one_hot], 1)
        x = mish(self.linear1(x))
        x = mish(self.linear2(x))
        x = mish(self.linear2_3(x))
        x = self.linear3(x)
        return x
class SACAgent:
    def __init__(self,state_dim,action_dim,hidden_dim):
        self.state_dim      = state_dim  
        self.action_dim     = action_dim 
        self.hidden_dim     = hidden_dim
        self.lr_pi          = 0.0001
        self.lr_q           = 0.0005
        self.lr_v           = 0.0005
        self.gamma          = 0.98
        self.batch_size     = 200
        self.buffer_limit   = 100000
        self.tau            = 0.0007   # for soft-update of Q using Q-target
        self.init_alpha     = 8
        self.target_entropy = -2 #-self.action_dim  # == -2
        self.lr_alpha       = 0.001
        self.loss           = [[0.,0.,0.,0.,0.]]
        self.DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory         = ReplayBuffer(self.buffer_limit, self.DEVICE)
        print("Device chosen : ", self.DEVICE)      
        self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(self.DEVICE)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)

        self.PI  = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim,self.lr_pi, self.DEVICE)
        self.Q1        = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim, self.lr_q, self.DEVICE)
        # self.Q1_target = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim, self.lr_q, self.DEVICE)
        self.Q2        = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim, self.lr_q, self.DEVICE)
        # self.Q2_target = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim, self.lr_q, self.DEVICE)

        self.V         = ValueNetwork(self.state_dim,self.hidden_dim,self.lr_v,self.DEVICE)
        self.V_target  = ValueNetwork(self.state_dim,self.hidden_dim,self.lr_v,self.DEVICE) 
        self.V_target.load_state_dict(self.V.state_dict())
        # self.Q1_target.load_state_dict(self.Q1.state_dict())
        # self.Q2_target.load_state_dict(self.Q2.state_dict())

    def choose_action(self, s):
        with torch.no_grad():
            action, log_prob = self.PI.sample(s.to(self.DEVICE))
        return action, log_prob

    def train_agent(self):
        mini_batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = mini_batch
        # torch.autograd.set_detect_anomaly(True)
        # td_target = self.calc_target(mini_batch)
        target_value=self.V_target(s_prime_batch)
        next_q_value = r_batch + self.gamma * (1 - done_batch) * target_value
        #### Q1 train ####
        q1_loss = F.smooth_l1_loss(self.Q1(s_batch, a_batch), next_q_value.detach())
        self.Q1.optimizer.zero_grad()
        q1_loss.mean().backward()
        nn.utils.clip_grad_norm_(self.Q1.parameters(), 1.0)
        self.Q1.optimizer.step()
        #### Q1 train ####
        #### Q2 train ####
        q2_loss = F.smooth_l1_loss(self.Q2(s_batch, a_batch), next_q_value.detach())
        self.Q2.optimizer.zero_grad()
        q2_loss.mean().backward()
        nn.utils.clip_grad_norm_(self.Q2.parameters(), 1.0)
        self.Q2.optimizer.step()
        #### Q2 train ####

        #### pi train ####
        a, log_prob = self.PI.sample(s_batch)
        # entropy = -self.log_alpha.exp() * log_prob

        q1, q2 = self.Q1(s_batch, a), self.Q2(s_batch, a)
        # q = torch.min(q1, q2).detach()  
        q = torch.min(q1, q2)
        # pi_loss = -(q + entropy)  # for gradient ascent
        pi_loss = (log_prob-q).detach()
        self.PI.optimizer.zero_grad()
        pi_loss.mean().backward()
        nn.utils.clip_grad_norm_(self.PI.parameters(), 1.0)
        self.PI.optimizer.step()
        #### pi train ####
        #### v train ####
        v_target=(q - log_prob).mean(dim=1, keepdim=True)
        v_loss=F.smooth_l1_loss(self.V(s_batch),v_target.detach())
        
        self.V.optimizer.zero_grad()
        v_loss.mean().backward()
        self.V.optimizer.step()
        #### v train ####

        #### alpha train ####
        # self.log_alpha_optimizer.zero_grad()
        # alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        # alpha_loss.backward()
        # self.log_alpha_optimizer.step()
        #### alpha train ####
        self.loss=[q1_loss.mean().detach().float().item(),q2_loss.mean().detach().float().item(),pi_loss.mean().detach().float().item(),\
                   v_loss.mean().detach().float().item()]
        #### V soft-update ####
        for param_target, param in zip(self.V_target.parameters(), self.V.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

class PathPlanningNode(Node):
    def __init__(self):
        super().__init__('path_planning_node')
        timer_period=0.5
        self.timer=self.create_timer(timer_period,self.training)
        # Publisher to control the robot
        
        self.current_position = [0, 0]

    def her_replay(self,env, episode_transitions, final_state):
        for transition in episode_transitions:
            s, a, r, s_prime, done = transition
            # Use final_state as a hindsight goal
            new_goal = final_state
            new_reward = env.get_reward(done, env.is_reach_goal(s_prime.position, self.neighbor_size))
            her_transition = (s, a, new_reward, s_prime, done)
            self.agent.memory.put(her_transition)
    def training(self):
        date = '27_11'
        # time.sleep(10)
        save_dir = "/home/mark/limo_ws/src/rl_sac/rl_sac/saved_model/path_planning" + date 
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        save_dir += "/"
        env=GridMap(debug=True)
        agent = SACAgent(STATE_DIMENSION,ACTION_DIMENSION,HIDDEN_SIZE) 
        self.get_logger().info(f'Device chosen: {agent.DEVICE}')
        # writer = SummaryWriter('SAC_log/'+date)
        score = 0.0
        print_once = True
        state_prime=None
        for EP in range(EPISODE):
            self.get_logger().info(f'EPISODE : {EP}')
            self.no_collision=0.
            self.no_reach_goal=0.
            env.generate_start_goal()
            state = env.reset()
            score, done = 0.0 , False        
            self.get_logger().info(f'Beginning : {EP}')
            episode_transitions = []

            for step in range(MAX_STEP_SIZE): #while not done:
                self.get_logger().info(f'STEP : {step}')

                action, _ = agent.choose_action(torch.FloatTensor(state))
                action = action.detach().cpu().numpy()
                state_prime, reward, done = env.step(action)
                transition = (state, action, reward, state_prime, done)
                episode_transitions.append(transition)

                agent.memory.put(transition)
                
                score += reward
                
                state = state_prime
                if done:
                    episode_transitions = []
                    self.get_logger().warn(f'COLLISION -> RESET')
                    state = env.reset()
                    self.no_collision+=1

                if agent.memory.size()>SAMPLE_THRESHOLD:
                    if print_once: 
                        self.get_logger().info('Start learning!.............................')
                        print_once = False
                    agent.train_agent()
                if agent.memory.size()<=SAMPLE_THRESHOLD:
                    # sim_rate.sleep()   
                    time.sleep(0.002)  
            final_state = state_prime
            goal=Point()
            goal.x=final_state[0]
            goal.y=final_state[1]
            for transition in episode_transitions:
                s, a, r, s_prime, done = transition
                done=False
                # Treat final state as the goal to create a new goal-based reward
                current_point=Point()
                current_point.x=s_prime[0]
                current_point.y=s_prime[1]
                new_reward = env.calculate_reward(current_point,goal)
                # Add HER transition
                agent.memory.put((s, a, new_reward, s_prime, done))

            print("EP:{}, Avg_Score:{:.1f}".format(EP, score), "\n")  
            if EP % 10 == 0: 
                torch.save(agent.PI.state_dict(), save_dir + "sac_actor_"+date+"_EP"+str(EP)+".pt")
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
def main(args=None):
    rclpy.init(args=args)
    node = PathPlanningNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
