#!/usr/bin/env python3

###ROS library
import copy
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
###ROS library
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import sys
import os
import shutil
from collections import deque, namedtuple
from std_msgs.msg import Float32MultiArray
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import GPUtil
import psutil
from threading import Thread
from csv import writer
from tensorboardX import SummaryWriter
# from std_srvs.srv import Empty
from .env_training import Env

action_dim = 2
state_dim  = 16
hidden_dim = 256
ACTION_V_MIN = 0.0 # m/s
ACTION_W_MIN = -2. # rad/s
ACTION_V_MAX = 0.22 # m/s
ACTION_W_MAX = 2. # rad/s
EPISODE, MAX_STEP_SIZE = 7000, 3000
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
        self.buffer_limit   = 500000
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

    # def calc_target(self, mini_batch):
    #     s, a, r, s_prime, done = mini_batch
    #     with torch.no_grad():
    #         a_prime, log_prob_prime = self.PI.sample(s_prime)
    #         entropy = - self.log_alpha.exp() * log_prob_prime
    #         q1_target, q2_target = self.Q1_target(s_prime, a_prime), self.Q2_target(s_prime, a_prime)
    #         q_target = torch.min(q1_target, q2_target)
    #         target = r + self.gamma * (1 - done) * (q_target + entropy)
    #     return target

    def train_agent(self):
        mini_batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = mini_batch
        torch.autograd.set_detect_anomaly(True)
        # td_target = self.calc_target(mini_batch)
        target_value=self.V_target(s_batch)
        next_q_value = r_batch + self.gamma * (1 - done_batch) * target_value
        #### Q1 train ####
        q1_loss = F.smooth_l1_loss(self.Q1(s_batch, a_batch), next_q_value)
        self.Q1.optimizer.zero_grad()
        q1_loss.mean().backward()
        nn.utils.clip_grad_norm_(self.Q1.parameters(), 1.0)
        self.Q1.optimizer.step()
        #### Q1 train ####
        next_q_value = next_q_value.detach()
        #### Q2 train ####
        q2_loss = F.smooth_l1_loss(self.Q2(s_batch, a_batch), next_q_value)
        self.Q2.optimizer.zero_grad()
        q2_loss.mean().backward()
        nn.utils.clip_grad_norm_(self.Q2.parameters(), 1.0)
        self.Q2.optimizer.step()
        #### Q2 train ####

        #### pi train ####
        a, log_prob = self.PI.sample(s_batch)
        entropy = -self.log_alpha.exp() * log_prob

        q1, q2 = self.Q1(s_batch, a), self.Q2(s_batch, a)
        # q = torch.min(q1, q2).detach()  
        q = torch.min(q1, q2)
        pi_loss = -(q + entropy)  # for gradient ascent
        self.PI.optimizer.zero_grad()
        pi_loss.mean().backward()
        nn.utils.clip_grad_norm_(self.PI.parameters(), 1.0)
        self.PI.optimizer.step()
        #### pi train ####
        #### v train ####
        v_target=(q - entropy).mean(dim=1, keepdim=True).detach()
        v_loss=F.smooth_l1_loss(self.V(s_batch),v_target)
        
        self.V.optimizer.zero_grad()
        v_loss.mean().backward()
        self.V.optimizer.step()
        #### v train ####

        #### alpha train ####
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        #### alpha train ####
        self.loss=[q1_loss.mean().detach().float().item(),q2_loss.mean().detach().float().item(),pi_loss.mean().detach().float().item(),\
                   v_loss.mean().detach().float().item(),alpha_loss.mean().detach().float().item()]
        #### V soft-update ####
        for param_target, param in zip(self.V_target.parameters(), self.V.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        #### Q1, Q2 soft-update ####
        # #### Q1, Q2 soft-update ####
        # for param_target, param in zip(self.V_target.parameters(), self.V.parameters()):
        #     param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        # for param_target, param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
        #     param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        # #### Q1, Q2 soft-update ####

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            print("CPU percentage:", psutil.cpu_percent())
            print('CPU virtual_memory used:', psutil.virtual_memory()[2], "\n")
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
class Training(Node):
    def __init__(self):
        super().__init__('mobile_robot_sac')
        timer_period=0.5 
        self.timer=self.create_timer(timer_period,self.training)
        self.velocity_publisher=self.create_publisher(Float32MultiArray,'vel_output',10)#[linear_vel,angular_vel]
        self.loss_publisher=self.create_publisher(Float32MultiArray,'loss',10)#[q1_loss,q2_loss,pi_loss,v_loss,alpha_loss]
        self.entropy_publisher=self.create_publisher(Float32MultiArray,'entropy',10)#[entropy_term]
        self.goal_distance_and_heading_publisher=self.create_publisher(Float32MultiArray,'goal_distance_and_heading',10)#[goal distance, heading]
        self.minimum_distance_from_obstacles=self.create_publisher(Float32MultiArray,'minimum_distance_obstacle',10)# min_distance_obstacle
        self.success_rate=self.create_publisher(Float32MultiArray,'success_rate',10) # [no_collision, no_reach_goal]success rate of reaching the goal without collisions
        self.reward_publisher=self.create_publisher(Float32MultiArray,'reward',10) #reward
    def training(self):
        GPU_CPU_monitor = Monitor(60)
        date = '10_11'
        # time.sleep(10)
        save_dir = "/home/mark/limo_ws/src/rl_sac/rl_sac/saved_model/" + date 
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        save_dir += "/"
        env = Env()
        agent = SACAgent(state_dim,action_dim,hidden_dim)
        # writer = SummaryWriter('SAC_log/'+date)
        score = 0.0
        print_once = True
        past_action = [0.,0.]
        with open('/home/mark/limo_ws/src/rl_sac/rl_sac/analysis/reward.csv', 'w', newline='') as csvfile:
            pass
        with open('/home/mark/limo_ws/src/rl_sac/rl_sac/analysis/success_rate.csv', 'w', newline='') as csvfile:
            pass
        for EP in range(EPISODE):
            self.get_logger().info(f'EPISODE : {EP}')
            self.no_collision=0.
            self.no_reach_goal=0.
            state,min_distance_obstacle,reach_goal = env.reset()
            self.get_logger().info(f"len(state):{len(state)} with  {state}")
            score, done = 0.0 , False        
            self.get_logger().info(f'Beginning : {EP}')
            for step in range(MAX_STEP_SIZE): #while not done:
                self.get_logger().info(f'STEP : {step}')

                ###Publish###
                msg=Float32MultiArray()
                msg.data=[float(EP),float(step),float(state[-1]),float(state[-2])]
                self.goal_distance_and_heading_publisher.publish(msg)
                msg.data=[float(EP),float(step),min_distance_obstacle]
                self.minimum_distance_from_obstacles.publish(msg)
                ###Publish###

                action, _ = agent.choose_action(torch.FloatTensor(state))
                action = action.detach().cpu().numpy()
                
                state_prime, reward, done,min_distance_obstacle,reach_goal = env.step(action, past_action,\
                                                     ACTION_V_MAX,ACTION_W_MAX)
                
                past_action = copy.deepcopy(action)
                agent.memory.put((state, action, reward, state_prime, done))
                
                score += reward
                
                state = state_prime
                ###Publish###
                msg.data=[float(EP),float(step),float(action[1]),float(action[0])]
                self.velocity_publisher.publish(msg)
                ###Publish###
                if done:
                    self.get_logger().warn(f'COLLISION -> RESET')
                    state,min_distance_obstacle,reach_goal = env.reset()
                    self.no_collision+=1

                if reach_goal:
                    self.no_reach_goal+=1
                if agent.memory.size()>10000:
                    if print_once: 
                        self.get_logger().info('Start learning!.............................')
                        print_once = False
                    agent.train_agent()
                    ###Publish###
                    msg.data=[float(EP),float(step)]+agent.loss
                    self.loss_publisher.publish(msg)
                    msg.data=[float(EP),float(step),agent.log_alpha.detach().item()]
                    self.entropy_publisher.publish(msg)
                    ###Publish###
                if agent.memory.size()<=10000:
                    # sim_rate.sleep()   
                    time.sleep(0.002)      
            with open('/home/mark/limo_ws/src/rl_sac/rl_sac/analysis/reward.csv', 'a', newline='') as csvfile:
                csvwriter = writer(csvfile, delimiter='|')
                csvwriter.writerow([float(EP),score])
                csvfile.close()
            with open('/home/mark/limo_ws/src/rl_sac/rl_sac/analysis/success_rate.csv', 'a', newline='') as csvfile:
                csvwriter = writer(csvfile, delimiter='|')
                csvwriter.writerow([float(EP),self.no_collision,self.no_reach_goal])
                csvfile.close()
            # writer.add_scalar("Score", score, EP)    
            print("EP:{}, Avg_Score:{:.1f}".format(EP, score), "\n")  
            #print("EP:{}, Avg_Score:{:.1f}, Q:{:.1f}, Entr:{:.1f}, Act_los:{:.1f}, Alpha:{:.3f}".format(n_epi, score, q/3000, ent/3000, actor_loss/3000, pi.log_alpha.exp()))
            #writer.add_scalar("Q_Value", q/3000, n_epi)
            #writer.add_scalar("Entropy", ent/3000 ,n_epi)
            #writer.add_scalar("Actor_Loss", actor_loss/3000 ,n_epi)
            #writer.add_scalar("alpha", alpha/3000 ,n_epi)
            
            #if pi.log_alpha.exp() > 1.0:
            #    pi.log_alpha += np.log(0.99)
            
            #if n_epi%print_interval==0 and n_epi!=0:
            #    print("# of episode :{}, avg score : {:.1f} alpha:{:.4f}".format(n_epi, score/print_interval, pi.log_alpha.exp()))
            #    writer.add_scalar("Score", score / print_interval, n_epi)
            #    score = 0.0
                
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
    try:
        rclpy.init(args=args)
        training = Training()
        rclpy.spin(training)
        training.destroy_node()
        rclpy.shutdown()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
