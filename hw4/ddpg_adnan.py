#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import copy
import shutil
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal

import gym
import pybullet
import pybulletgym.envs

from collections import deque
from operator import itemgetter
from statistics import mean
from tqdm import tqdm

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='DDPG')
parser.add_argument('-s', '--seed', type=int, help='seed for random initializations')
parser.add_argument('-i', '--index', type=int, help='index of run')

def weighSync(target_model, source_model, tau=0.001):
    ''' A function to soft update target networks '''
    assert isinstance(tau, float) and tau>0

    for param_target, param_source in zip(target_model.parameters(), source_model.parameters()):
        # Wrap in torch.no_grad() because weights have requires_grad=True, 
        # but we don't need to track this in autograd
        with torch.no_grad():
            param_target = tau*param_source + (1-tau)*param_target
    
    return target_model, source_model


class Replay():
    def __init__(self, buffer_size, init_length, state_dim, action_dim, env):
        """
        A function to initialize the replay buffer.

        param: init_length : Initial number of transitions to collect
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        param: env : gym environment object
        """
        assert isinstance(buffer_size, int) and buffer_size>0
        assert isinstance(init_length, int) and init_length>0
        assert isinstance(state_dim, int) and state_dim>0
        assert isinstance(action_dim, int) and action_dim>0
        
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.buffer = deque() #list like object for which removing elements from left is faster
        
        s = env.reset()
        for i in range(init_length):
            a = env.action_space.sample()
            s_prime, r, done, _ = env.step(a)
            self.buffer.append((s,a,r,s_prime))
    
    def __len__(self):
        ''' Return number of elements in buffer'''
        return len(self.buffer)

    def buffer_add(self, exp):
        """
        A function to add a dictionary to the buffer
        param: exp : A tuple consisting of (state, action, reward, next state) in that order
        """
        assert isinstance(exp, tuple) and len(exp) == 4
        assert len(self.buffer) <= self.buffer_size, 'Buffer size exceeded. You fucked up'
        
        if len(self) < self.buffer_size:
            self.buffer.append(exp)
        else:
            self.buffer.popleft() #removing the 1st element (left most element)
            self.buffer.append(exp)

    def buffer_sample(self, N):
        """
        A function to sample N points from the buffer
        param: N : Number of samples to obtain from the buffer
        """
        assert isinstance(N, int) and N>0
        indices = list(np.random.randint(low=0, high=len(self), size=N, dtype='int'))
        sample = itemgetter(*indices)(self.buffer) #extarct values at indices from buffer
        
        assert len(sample) == N, 'You fucked up sampling bruh'
        return sample


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Initialize the network
        param: state_dim : Size of the state space
        param: action_dim: Size of the action space
        """
        assert isinstance(state_dim, int) and state_dim>0
        assert isinstance(action_dim, int) and action_dim>0
        
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # NN layers and activations
        self.fc1 = nn.Linear(state_dim, 400)
        self.hidden1 = nn.Linear(400, 300)
        self.fc2 = nn.Linear(300, action_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, state):
        """
        Define the forward pass
        param: state: The state of the environment
        """
        if not(isinstance(state, torch.Tensor)):
            state = torch.from_numpy(state).float()
        
        x = state
        x = self.relu(self.fc1(x))
        x = self.relu(self.hidden1(x))
        x = self.tanh(self.fc2(x))

        return x
    
    def getAction(self, state, add_noise_flag = False, noise = 0.1):
        '''
        Returns an action by doing a forward pass. If add_noise_flag is True, 
        action is sampled from a multivariate Normal distributio with stddev = noise and mean = output of net
        
        :rtype: np.ndarray
        '''
        assert isinstance(state, np.ndarray)
        assert isinstance(noise, (int, float)) and noise >=0
        assert isinstance(add_noise_flag, bool)
        
        state = torch.from_numpy(state).float()
        action = self.forward(state) #forward pass
        
        if add_noise_flag:
            # Sampling from the nD Gaussian
            m = MultivariateNormal(action, torch.eye(self.action_dim)*noise)
            action = m.sample()
        
        return action.detach().squeeze().numpy()


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Initialize the critic
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        """
        assert isinstance(state_dim, int) and state_dim>0
        assert isinstance(action_dim, int) and action_dim>0
        
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # NN layers and activations
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.hidden1 = nn.Linear(400, 300)
        self.fc2 = nn.Linear(300, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        """ Define the forward pass of the critic """
        assert isinstance(state, np.ndarray)
        # assert isinstance(action, np.ndarray)
        assert state.shape[1] == self.state_dim, 'state must be of dim (batch_size, %d)'%self.state_dim
        # assert action.shape[1] == self.action_dim, 'action must be of dim (batch_size, %d)'%self.action_dim
        
        state = torch.from_numpy(state).float(),  #numpy to torch tensor
        
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()

        try:
            x = torch.cat((state, action), dim=0) #concatenating to form input
        except RuntimeError:
            x = torch.cat((state, action), dim=1) #concatenating to form input
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.hidden1(x))
        x = self.fc2(x)
        
        return x


class DDPG():
    def __init__(
            self,
            env,
            test_env,
            state_dim,    
            action_dim,
            critic_lr=3e-4,
            actor_lr=3e-4,
            gamma=0.99,
            batch_size=100,
            ev_n_steps=100,
            verbose=False
    ):
        """
        Implementing DPPG algorithm from paper - Continuous control with deep reinforcement learning
        link - https://arxiv.org/pdf/1509.02971.pdf
        
        param: env: An gym environment
        param: action_dim: Size of action space
        param: state_dim: Size of state space
        param: critic_lr: Learning rate of the critic
        param: actor_lr: Learning rate of the actor
        param: gamma: The discount factor
        param: batch_size: The batch size for training
        """

        assert isinstance(state_dim, int) and state_dim>0
        assert isinstance(action_dim, int) and action_dim>0
        assert isinstance(batch_size, int) and batch_size>0
        assert isinstance(critic_lr, (int, float)) and critic_lr>0
        assert isinstance(actor_lr, (int, float)) and actor_lr>0
        assert isinstance(gamma, (int, float)) and gamma>0
        assert isinstance(ev_n_steps, (int, float)) and ev_n_steps>0

        self.gamma = gamma
        self.batch_size = batch_size
        self.ev_n_steps = ev_n_steps
        self.env = env
        self.test_env = test_env
        self.num_episodes = 0
        self.avg_rewards = []
        self.obj_actor = []
        self.loss_critic = []

        # Create a actor and actor_target with same initial weights
        self.actor = Actor(state_dim, action_dim)
        self.actor = self.init_weights(self.actor) #initialize weights according to paper
        self.actor_target = copy.deepcopy(self.actor) #both networks have the same initial weights 

        # Create a critic and critic_target with same initial weights
        self.critic = Critic(state_dim, action_dim)
        self.critic = self.init_weights(self.critic) #initialize weights according to paper
        self.critic_target = copy.deepcopy(self.critic) #both networks have the same initial weights 

        # Define optimizer for actor and critic
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.loss_fn = nn.MSELoss(reduction='mean')

        # Define a replay buffer
        self.ReplayBuffer = Replay(10000, 1000, state_dim, action_dim, self.env)
    
    def init_weights(self, network):
        '''
        Initialize weights (as mentioned in paper) from a uniform distribution 
        based on the fan-in of the layer
        
        WARNING: Will only work if each layer is fully connected
        '''
        
        for name, param in network.named_parameters():
            if 'bias' in name: #if bias param, use f of the same layer
                f = last_f
            else:
                f = param.shape[1] #picking 2nd dim = number of inputs for that layer
                last_f = f
            # Initialize weights by sampling from uniform dist.
            assert isinstance(f, int) and f>0, 'fan in must be int and greater than 0'
            nn.init.uniform_(param.data, a = -1/np.sqrt(f), b = 1/np.sqrt(f))
        
        return network
    
    def save_actor(self, index):
        ''' Saves the policy NN'''
        filename = 'q1_policy_' + str(index) + '.pth.tar'
        state = {'state_dict': self.actor.state_dict(),
                 'optimizer': self.optimizer_actor.state_dict(),
                 'seed': args.seed }
        torch.save(state, filename)
        torch.save(self.best_actor_state, 'best_'+filename)
    
    def update_target_networks(self):
        """
        A function to update the target networks
        """
        self.actor_target, self.actor = weighSync(self.actor_target, self.actor)
        self.critic_target, self.critic = weighSync(self.critic_target, self.critic)
    
    def getAverageReward(self):
        ''' Run the policy and return average reward '''
        rewards = []
        s = self.test_env.reset()
        done = False
        while not(done):
            a = self.actor(s).detach().squeeze().numpy()
            s, r, done, _ = self.test_env.step(a)
            rewards.append(r)
        
        avg_reward = sum(rewards)
        assert isinstance(avg_reward, (int, float))
        
        return avg_reward

    def train(self, max_num_steps):
        """
        Train the policy for the given number of iterations
        :param num_steps:The number of steps to train the policy for
        """
        assert isinstance(max_num_steps, int) and max_num_steps>0
        
        gamma = self.gamma
        state = self.env.reset()
        done = False
        best_r = -np.inf
        # Training starts now
        for num_steps in tqdm(range(max_num_steps)):
            # Reset env when it reached terminal state
            if done:
                self.num_episodes += 1
                state = self.env.reset()
            
            action = self.actor.getAction(state, add_noise_flag = True, noise = 0.1)
            state_next, rewd, done, _ = self.env.step(action)
            
            # Storing transition in buffer
            self.ReplayBuffer.buffer_add((state,action,rewd,state_next))
            state = state_next

            # Sampling N points from buffer
            minibatch = self.ReplayBuffer.buffer_sample(N = self.batch_size)

            
            # Operating on minibatch
            s = np.array([el[0] for el in minibatch])                        #dim (batch_size, state_dim)
            a = np.array([el[1] for el in minibatch])                        #dim (batch_size, action_dim)
            r = torch.Tensor([el[2] for el in minibatch]).unsqueeze(dim=1)   #dim (batch_size, 1)
            s_prime = np.array([el[3] for el in minibatch])                  #dim (batch_size, state_dim)
            
            a_target = self.actor_target.getAction(s_prime, add_noise_flag=False) #dim(batch_size, action_dim)
            assert s_prime.shape[0] == a_target.shape[0], 'First dim should be batch_size'

            print('type(s_prime), s_prime = ', type(s_prime), s_prime)
            y_i = r + gamma*self.critic_target(s_prime, a_target)            #dim (batch_size, 1)
            loss_critic = self.loss_fn(self.critic(s,a), y_i) #mse loss      #dim (batch_size, 1)
            
            a = self.actor(s)
            # a = self.actor.getAction(s, add_noise_flag=False)
            obj_actor = self.critic(s, a).mean()
            
            
            # Update critic
            loss_critic.backward()
            self.optimizer_critic.step()
            # Update actor
            obj_actor *= -1 #multiplying with negative so it does gradient ascent
            obj_actor.backward()
            self.optimizer_actor.step()
            # Update target networks
            self.update_target_networks()
            
            # Store losses
            self.obj_actor.append(obj_actor.item())
            self.loss_critic.append(loss_critic.item())

            # Zero gradients of optimizer
            self.optimizer_critic.zero_grad()    
            self.optimizer_actor.zero_grad()
            
            if num_steps%self.ev_n_steps == 0:
                r = self.getAverageReward()
                # Saving best actor model till now
                is_best = r > best_r
                best_r = max(r, best_r)
                if is_best:
                    self.best_actor_state = {'state_dict': self.actor.state_dict(),
                                             'optimizer': self.optimizer_actor.state_dict(),
                                             'avg_reward': r,
                                             'obj_actor': obj_actor.item(),
                                             'loss_critic': loss_critic.item(),
                                             'seed': args.seed
                                            }
                
                self.avg_rewards.append(r)
                if verbose:
                    print('Num steps: {0} \t Avg Reward: {1:.3f} \t Obj(Actor): {2:.3f} \t Loss(Critic): {3:.3f} \t Num eps: {4}'
                          .format(num_steps, r, obj_actor.item(), loss_critic.item(), self.num_episodes))
                


if __name__ == "__main__":
    global args
    args = parser.parse_args()
    
    # Seed value
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Define the environment
    env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=True)
    test_env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)
    plot = True
    verbose = True
    
    # Define Deep Deterministic Policy Gradient object
    ddpg_object = DDPG(
        env,
        test_env,
        8,
        2,
        critic_lr=1e-3,
        actor_lr=1e-4,
        gamma=0.99,
        batch_size=500,
        ev_n_steps=200, #evaluate every n steps
        verbose=verbose
    )
    # Train the policy
    ddpg_object.train(int(5e5))
    
    # Save actor
    ddpg_object.save_actor(args.index)
    
    if plot:
        fig, axs = plt.subplots(3,1, figsize=(10,15))
        axs = axs.flatten()
        axs[0].plot(ddpg_object.avg_rewards)
        axs[0].set_ylabel('Avg Rewards')
        axs[0].set_xlabel('Every %d iterations'%ddpg_object.ev_n_steps)
        axs[0].set_title(' Avg Rewards vs iterations ')
    #     axs[1].plot(ddpg_object.obj_actor[::ddpg_object.ev_n_steps])
        axs[1].plot(ddpg_object.obj_actor)
        axs[1].set_ylabel('Actor Obj')
        axs[1].set_title(' Actor Obj vs iterations ')
    #     axs[2].plot(ddpg_object.loss_critic[::ddpg_object.ev_n_steps])
        axs[2].plot(ddpg_object.loss_critic)
        axs[2].set_ylabel('Critic Loss')
        axs[2].set_title(' Critic Loss vs iterations ')
        plt.savefig('index_'+str(args.index)+'.png', dpi=300)

