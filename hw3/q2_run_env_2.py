import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import gym
import time
import pybulletgym  # register PyBullet enviroments with open ai gym

class Policy(nn.Module):

    def __init__(self, num_classes, lr):
        super(Policy, self).__init__()
        
        self.sigma = nn.parameter.Parameter(torch.tensor([[0.1, 0.], [0., 0.1]]))
        
        self.fc1 = nn.Linear(8, 64)
        self.hidden = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.tanh = nn.Tanh()
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, x):
        
        x = self.fc1(x) #input
        x = self.tanh(x)
        x = self.hidden(x) #layer 1
        x = self.tanh(x)
        x = self.hidden(x) #layer 2
        x = self.tanh(x)
        x = self.fc2(x) #layer 3
        x = self.tanh(x)

        sigma = torch.relu(self.sigma) + torch.tensor([[0.001, 0.], [0., 0.001]])

        return x, sigma

def makePolicyNN(num_actions=2, lr=0.01):
    ''' Initialize the policy class '''
    assert isinstance(num_actions, int) and num_actions>0
    
    return Policy(num_actions, lr)

def getAction(self, policy_network, state):
    ''' Return an action from a stochastic policy '''
    assert isinstance(state, np.ndarray)
    
    state = torch.from_numpy(state).float()
    torque_mean, torque_sigma = policy_network(state) #forward pass
    
    # Sampling from the 2D Gaussian and calculating the actions log probability
    m = MultivariateNormal(torque_mean, torque_sigma)
    action = m.sample() #type tensor
    log_prob_of_action = m.log_prob(action) #type tensor

    return action, log_prob_of_action

if __name__ == '__main__':

    policy_network_file = './' + 'q2_policy4' + '.pth.tar'
    policy_network = makePolicyNN(num_actions=2, lr=1e-3)

    print(' =======> Loading model from file ', policy_network_file)
    network_state_dict = torch.load(policy_network_file)
    policy_network.load_state_dict(network_state_dict['state_dict'])
    policy_network.optimizer.load_state_dict(network_state_dict['optimizer'])
    print(' =======> Model loaded. Rendering Environment now')

    env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)
    steps = 0
    env.render('human')
    state = env.reset()
    done = False
    time.sleep(3)
    while not(done):
        state = torch.from_numpy(state).float()
        a, _ = policy_network(state)
        state, r, done, info = env.step(np.array(a.data))
        steps+=1
        env.render('human')
        time.sleep(0.2)
    print('Total steps = ', steps)
    env.env.close()

    # env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)
    # steps = 0
    # env.render('human')
    # state = env.reset()
    # done = False
    # time.sleep(1)
    # while steps<200:
    #     a, _ = policy_network(state) #forward pass
    #     state, r, done, info = env.step(np.array(a))
    #     steps+=1
    #     env.render('human')
    #     time.sleep(0.05)
    # print('Total steps = ', steps)
    # env.env.close()

