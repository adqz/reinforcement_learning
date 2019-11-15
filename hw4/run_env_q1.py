import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import gym
import time
import pybulletgym  # register PyBullet enviroments with open ai gym

class Actor(nn.Module):
    #TODO: Complete the function
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
#         assert isinstance(state, torch.Tensor)
        assert state.shape == (self.state_dim, ), 'state must be 1D and of size (%d,)'%self.state_dim
        
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
#         print('type(state), shape = ',type(state), state.shape)
#         print('self.state_dim = ', self.state_dim)
        assert isinstance(state, np.ndarray)
        assert state.shape == (self.state_dim, )
        assert isinstance(noise, (int, float)) and noise >=0
        assert isinstance(add_noise_flag, bool)
        
        state = torch.from_numpy(state).float()
        action = self.forward(state) #forward pass
        
        if add_noise_flag:
            # Sampling from the nD Gaussian
            m = MultivariateNormal(action, torch.eye(self.action_dim)*noise)
            action = m.sample()
        
        return action.detach().squeeze().numpy()

if __name__ == '__main__':

    state_dim, action_dim = 8, 2    
    policy_network_file = './q1_policy_5' + '.pth.tar'
    policy_network = Actor(state_dim, action_dim)

    print(' =======> Loading model from file ', policy_network_file)
    network_state_dict = torch.load(policy_network_file)
    policy_network.load_state_dict(network_state_dict['state_dict'])
    try:
    print('Stats when saved - Avg Reward: {0:.3f} \t Obj(Actor): {1:.3f} \t Loss(Critic): {2:.3f}'\
        .format(network_state_dict['avg_reward'],network_state_dict['obj_actor'], network_state_dict['loss_critic']))
    except KeyError:
        pass
    print(' =======> Model loaded. Rendering Environment now')
    

    env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)
    steps = 0
    env.render('human')
    state = env.reset()
    done = False
    time.sleep(1)
    while not(done):
        a = policy_network(state).detach().squeeze().numpy()
        state, r, done, info = env.step(a)
        steps+=1
        env.render('human')
        time.sleep(0.1)
    print('Total steps = ', steps)
    env.env.close()
