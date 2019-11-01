#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import numpy as np
from numpy import linalg as LA

np.set_printoptions(precision=4, suppress=True)


class Conversions():

    def R3Toso3(self, w):
        ''' Convert from R3 to so3'''
        assert isinstance(w, np.ndarray)
        assert w.size == 3
        w = w.reshape((-1,)) #converting to size(3,)
        so3 = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0],
        ])
        return so3
    
    def so3ToR3(self, so3):
        ''' Convert from so3 to R3'''
        assert isinstance(so3, np.ndarray)
        assert so3.shape == (3,3)
        
        return np.array([so3[2,1], so3[0,2], so3[1,0]])
    
    def R6Tose3(self, S):
        '''Convert from R6 to se3'''
        assert isinstance(S, np.ndarray)
        assert S.size == 6
        
        S = S.reshape((-1,)) #converting to size(3,)
        w, v = S[0:3], S[3:]
        se3 = np.zeros((4,4))
        se3[0:3, 0:3] = self.R3Toso3(w)
        se3[0:3, 3] = v
        return se3
    
    def se3ToR6(self, V):
        assert isinstance(V, np.ndarray)
        assert V.shape == (4,4)
        
        w = self.so3ToR3(V[0:3, 0:3])
        v = V[0:3, 3]
        V_R6 = np.hstack((w, v))
        
        assert V_R6.shape == (6,)
        return V_R6
    
    def se3ToSE3(self, S, theta):
        '''
        Convert from se3 to SE3
        Note: R and p formula from pg 113, Ch 3, Modern robotics mechanics, planning, control by Kevin Lynch
        
        :param S: Screw axis of joint
        :type S: numpy array
        :param theta: angle of joint
        :type theta: float or int
        '''
        assert isinstance(S, np.ndarray)
        assert isinstance(theta, float) or isinstance(theta, int)
        assert S.size == 6
        
        S = S.reshape((-1,)) #converting to size(6,)
        w, v = S[0:3], S[3:] #extracting w,v from Screw vector
        w_hat = self.R3Toso3(w)
        R = np.eye(3) + np.sin(theta)*w_hat + (1 - np.cos(theta))*w_hat@w_hat
        p = ((theta*np.eye(3)) +              ((1 - np.cos(theta))*w_hat) +              ((theta - np.sin(theta))*w_hat@w_hat)) @ v
        SE3 = np.zeros((4,4))
        SE3[0:3,0:3] = R
        SE3[0:3,3] = p
        SE3[3,3] = 1.0
        return SE3
    
    def getAdjunct(self, T):
        '''
        Calculates the adjunct matrix for a homogeneous matrix T which belongs to SE3
        
        :param T: A homogeneous matrix belonging to SE3
        :type T: numpy array, 4x4
        :rtype: numpy array, 6x6
        '''
        R, p = T[0:3,0:3], T[0:3,3]
        p_hat = self.R3Toso3(p)
        AdjunctT = np.zeros((6,6))
        AdjunctT[0:3,0:3] = R
        AdjunctT[3:, 0:3] = p_hat @ R
        AdjunctT[3:,3:] = R
        
        assert AdjunctT.shape == (6,6)
        return AdjunctT
    
    def transToRp(self, T):
        '''
        Returns the Rotation matrix and position vector contained in the homogeneous transformation matrix T
        
        :param T: A homogeneous matrix belonging to SE3
        :type T: numpy array, 4x4
        '''
        assert isinstance(T, np.ndarray)
        assert T.shape == (4,4)
        
        return T[0:3, 0:3], T[0:3, 3]
    
    def invOfTrans(self, T):
        '''
        Returns inverse of the homogeneous transformation matrix T
        
        :param T: A homogeneous matrix belonging to SE3
        :type T: numpy array, 4x4
        '''
        assert isinstance(T, np.ndarray)
        assert T.shape == (4,4)
        
        R, p = self.transToRp(T)
        invT = np.zeros((4,4))
        invT[0:3, 0:3] = R.T
        invT[0:3, 3] = -(R.T @ p)
        invT[3, 3] = 1
        
        assert invT.shape == (4,4)
        return invT

    def nearZero(self, z):
        """Determines whether a scalar is small enough to be treated as zero"""
        return abs(z) < 1e-6
    
    def matrixLogOfR(self, R):
        acosinput = (np.trace(R) - 1) / 2.0
        if acosinput >= 1:
            return np.zeros((3, 3))
        elif acosinput <= -1:
            if not self.nearZero(1 + R[2][2]):
                omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) * np.array([R[0][2], R[1][2], 1 + R[2][2]])
            elif not self.nearZero(1 + R[1][1]):
                omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) * np.array([R[0][1], 1 + R[1][1], R[2][1]])
            else:
                omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) * np.array([1 + R[0][0], R[1][0], R[2][0]])
            return self.R3Toso3(np.pi * omg)
        else:
            theta = np.arccos(acosinput)
            return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)

    def matrixLogOfT(self, T):
        
        R, p = self.transToRp(T)
        omgmat = self.matrixLogOfR(R)
        if np.array_equal(omgmat, np.zeros((3, 3))):
            return np.r_[np.c_[np.zeros((3, 3)),
                               [T[0][3], T[1][3], T[2][3]]],
                         [[0, 0, 0, 0]]]
        else:
            theta = np.arccos((np.trace(R) - 1) / 2.0)
            return np.r_[np.c_[omgmat,
                               np.dot(np.eye(3) - omgmat / 2.0 \
                               + (1.0 / theta - 1.0 / np.tan(theta / 2.0) / 2) \
                                  * np.dot(omgmat,omgmat) / theta,[T[0][3],
                                                                   T[1][3],
                                                                   T[2][3]])],
                         [[0, 0, 0, 0]]]


def getForwardModel(thetalist):
    '''
    For a 2-dof arm, this function takes joint states (q0,q1) as input and returns the end effector position.
    Units for q0, q1 are radians.
    Forward kinematics is done in world/space frame
    
    :param thetalist: list containing 2 joint angles
    :type thetalist: list
    :rtype: numpy array, 4x4
    '''
    assert isinstance(thetalist, list) and len(thetalist) == 2
    assert all([isinstance(q, (int,float)) for q in thetalist])
    
    l0, l1 = 0.1, 0.11
    M = np.array([
        [1, 0, 0, l0+l1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    S = np.array([[0, 0, 1, 0, 0, 0],
                  [0, 0, 1, 0, -l0, 0],
                 ]).T
    f = Conversions()
    
    # Forward Kinematics in world/space frame
    T = np.array(M)
    for i in range(len(thetalist)-1,-1,-1):
        exp_S_theta = f.se3ToSE3(S[:,i], thetalist[i])
        T = exp_S_theta @ T
    return T

def getJacobian(thetalist):
    '''
    For a 2-dof arm, this function takes joint states (q0,q1) as input and returns the Jacobian.
    Units for q0, q1 are radians.
    Jacobian calculated in space frame/world frame
    
    :param thetalist: list containing 2 joint angles
    :type thetalist: list
    :rtype: numpy array, 6x6
    '''
    assert isinstance(thetalist, list) and len(thetalist) == 2
    assert all([isinstance(q, (int,float)) for q in thetalist])

    l0, l1 = 0.1, 0.11
    S = np.array([[0, 0, 1, 0, 0, 0],
                  [0, 0, 1, 0, -l0, 0],
                 ]).T
    J = np.zeros((6, len(thetalist)))
    T = np.eye(4)
    f = Conversions()
    for i in range(len(thetalist)):
        exp_S_theta = f.se3ToSE3(S[:,i], thetalist[i]) # size(4,4)
        T = T @ exp_S_theta # size(4,4)
        Adj_Si = f.getAdjunct(T) # size(6,6)
        J[:,i] = Adj_Si @ S[:,i] # size(6,6) x size(6,1) = size(6,1)
    return J

def getIK(x_d, y_d, theta_d, initial_guess = np.array([0.3, 0.4]), iterative = True):
    '''
    Implements inverse kinematics for the 2-dof arm using an analytical solution
    Ref: https://modernrobotics.northwestern.edu/nu-gm-book-resource/inverse-kinematics-of-open-chains/#department

    :param T: A homogeneous matrix belonging to SE3
    :type T: numpy array, 4x4
    :rtype: list of joint angles 
    '''
    assert isinstance(x_d, int) or isinstance(x_d, float)
    assert isinstance(y_d, int) or isinstance(y_d, float)
    assert isinstance(theta_d, int) or isinstance(theta_d, float)
    assert isinstance(initial_guess, np.ndarray) and len(initial_guess) == 2
    assert isinstance(iterative, bool)
    
    eomg, ev = 1e-3, 1e-3;
    i, maxIter = 0, 20
    
    l0, l1 = 0.1, 0.11    
    T_sd = np.array([
        [np.cos(theta_d), -np.sin(theta_d), 0, x_d],
        [np.sin(theta_d), np.cos(theta_d), 0, y_d],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]) #Standard form of Transformation matrix for rotation over z axis
    
    f = Conversions()
    theta = initial_guess.copy()
    T_sb = getForwardModel(list(theta)) #Forward kinematics with current solution
    # Calculating desired twist in space/world frame
    T_bs = f.invOfTrans(T_sb) # T_bs = inv(T_sb)
    T_bd = T_bs @ T_sd # T_bd = T_bs x T_sd
    V_b = f.matrixLogOfT(T_bd) # [V_b] belong to se3
    V_b = f.se3ToR6(V_b) #desired twist in body frame
    J_s = getJacobian(list(theta)) #Jacobian in world/space frame
    assert V_b[0] == 0.0 and V_b[1] == 0.0, 'There should be no angular velocity in x and y direction'
    assert V_b[5] == 0.0 , 'There should be no linear velocity in z direction'
    V_s = f.getAdjunct(T_sb) @ V_b #desired twist in world/space frame
    w, v = V_s[0:3], V_s[3:]
    err = LA.norm(w) > eomg or LA.norm(v) > ev
    
    while(i<maxIter and err):
        theta += LA.pinv(J_s) @ V_s
        if not(iterative): # Calculating angles in one go
            return theta, True
        # Calculating desired twist in space/world frame
        T_sb = getForwardModel(list(theta))
        T_bs = f.invOfTrans(T_sb) 
        T_bd = T_bs @ T_sd 
        V_b = f.matrixLogOfT(T_bd)
        V_b = f.se3ToR6(V_b) #desired twist in body frame
        assert V_b[0] == 0.0 and V_b[1] == 0.0, 'There should be no angular velocity in x and y direction'
        assert V_b[5] == 0.0 , 'There should be no linear velocity in z direction'
        V_s = f.getAdjunct(T_sb) @ V_b #desired twist in world/space frame
        w, v = V_s[0:3], V_s[3:]
        err = LA.norm(w) > eomg or LA.norm(v) > ev
        i+=1
    return theta, not(err)    

def getState():
    q0, q0_dot = env.unwrapped.robot.central_joint.current_position()
    q1, q1_dot = env.unwrapped.robot.elbow_joint.current_position()
    return [q0,q1], [q0_dot, q1_dot]

def setJointAngles(thetalist):
    assert isinstance(thetalist, list) and len(thetalist) == 2
    assert all([isinstance(q, (int,float)) for q in thetalist])
    
    env.unwrapped.robot.central_joint.reset_position(thetalist[0], 0)
    env.unwrapped.robot.robot_elbom.reset_position(thetalist[1], 0)
    return

if __name__ == '__main__':
    env = gym.make("ReacherPyBulletEnv-v0")
    env._max_episode_steps = 1000
    env.render()
    env.reset()
    for i in range(0,1000000):
        setJointAngles([0, 0])
        env.render()

    # theta, theta_dot = getState()
    # print(theta, theta_dot)
