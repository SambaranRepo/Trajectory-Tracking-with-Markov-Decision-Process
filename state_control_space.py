import numpy as np 
import pickle
import itertools as it

def generate_state_space(): 
    '''
    : function to create the state space of the tracking problem 
    : the state space contains time, error_x, error_y and error_th
    '''
    t = np.arange(0,51,0.5)[:-1]
    e_x= [-3, -0.5] + list(np.linspace(-0.25, 0.25, 5)) + [0.5,3]
    e_y = e_x
    th = [-np.pi, -np.pi/2] + list(np.linspace(-np.pi/4, np.pi/4, 5)) + [np.pi/2, np.pi] 
    X = list(it.product(t,e_x,e_y, th))
    state_table = {}
    n_states = len(X)
    for i in range(n_states):
        state_table.update({X[i]:i})

    return X, e_x, e_y, th, state_table

def generate_control_space():
    '''
    : function to generate the control space for the tracking problem 
    : control includes the linear velocity and the angular velocity of the robot
    '''
    v = np.linspace(0,1,5)
    w = np.linspace(-1,1,10)
    U = list(it.product(v,w))
    return U

