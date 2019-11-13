""" Learn a policy using DDPG for the reach task"""
import numpy as np
import torch
import time
import torch.nn as nn

import gym
import pybullet
import pybulletgym.envs

import matplotlib.pyplot as plt

np.random.seed(1000)


# TODO: A function to soft update target networks
def weighSync(target_model, source_model, tau=0.001):
    raise NotImplementedError


# TODO: Write the ReplayBuffer
class Replay():
    def __init__(self, buffer_size, init_length, state_dim, action_dim, env):
        """
        A function to initialize the replay buffer.

        param: init_length : Initial number of transitions to collect
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        param: env : gym environment object
        """
        raise NotImplementedError

    # TODO: Complete the function
    def buffer_add(self, exp):
        """
        A function to add a dictionary to the buffer
        param: exp : A dictionary consisting of state, action, reward , next state and done flag
        """
        raise NotImplementedError

    #TODO: Complete the function
    def buffer_sample(self, N):
        """
        A function to sample N points from the buffer
        param: N : Number of samples to obtain from the buffer
        """
        raise NotImplementedError


# TODO: Define an Actor
class Actor(nn.Module):
    #TODO: Complete the function
    def __init__(self, state_dim, action_dim):
        """
        Initialize the network
        param: state_dim : Size of the state space
        param: action_dim: Size of the action space
        """
        raise NotImplementedError

    #TODO: Complete the function
    def forward(self, state):
        """
        Define the forward pass
        param: state: The state of the environment
        """
        raise NotImplementedError


# TODO: Define the Critic
class Critic(nn.Module):
    # TODO: Complete the function
    def __init__(self, state_dim, action_dim):
        """
        Initialize the critic
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        """
        raise NotImplementedError

    # TODO: Complete the function
    def forward(self, state, action):
        """
        Define the forward pass of the critic
        """
        raise NotImplementedError


# TODO: Implement a DDPG class
class DDPG():
    def __init__(
            self,
            env,
            action_dim,
            state_dim,
            critic_lr=3e-4,
            actor_lr=3e-4,
            gamma=0.99,
            batch_size=100,
    ):
        """
        param: env: An gym environment
        param: action_dim: Size of action space
        param: state_dim: Size of state space
        param: critic_lr: Learning rate of the critic
        param: actor_lr: Learning rate of the actor
        param: gamma: The discount factor
        param: batch_size: The batch size for training
        """
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = env

        # TODO: Create a actor and actor_target
        self.actor = None
        self.actor_target = None
        # TODO: Make sure that both networks have the same initial weights

        # TODO: Create a critic and critic_target object
        self.critic = None
        self.critic_target = None
        # TODO: Make sure that both networks have the same initial weights

        # TODO: Define the optimizer for the actor
        self.optimizer_actor = None
        # TODO: Define the optimizer for the critic
        self.optimizer_critic = None

        # TODO: define a replay buffer
        self.ReplayBuffer = None

    # TODO: Complete the function
    def update_target_networks(self):
        """
        A function to update the target networks
        """
        weighSync(self.actor_target, self.actor)
        weighSync(self.critic_target, self.critic)

    # TODO: Complete the function
    def update_network(self):
        """
        A function to update the function just once
        """
        raise NotImplementedError

    # TODO: Complete the function
    def train(self, num_steps):
        """
        Train the policy for the given number of iterations
        :param num_steps:The number of steps to train the policy for
        """
        raise NotImplementedError


if __name__ == "__main__":
    # Define the environment
    env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)

    ddpg_object = DDPG(
        env,
        8,
        2,
        critic_lr=1e-3,
        actor_lr=1e-3,
        gamma=0.99,
        batch_size=100,
    )
    # Train the policy
    ddpg_object.train(100)

    # Evaluate the final policy
    state = env.reset()
    done = False
    while not done:
        action = ddpg_object.actor(state).detach().squeeze().numpy()
        next_state, r, done, _ = env.step(action)
        env.render()
        time.sleep(0.1)
        state = next_state
