from tabnanny import check
import numpy as np
import pickle
from tqdm import tqdm
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, thread
import asyncio
import itertools as it
from state_control_space import *

with open('MDP.pkl', 'rb') as f:
    x = pickle.load(f)
    P = x[0]

executor = ThreadPoolExecutor()
def threadpool(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        return asyncio.wrap_future(executor.submit(f,*args, **kwargs))
    return wrap

X, e_x, e_y, th, state_table = generate_state_space()
U = generate_control_space()
n_states = len(X)
n_controlspace = len(U)

time_step = 0.5
def lissajous(k):
    xref_start = 0
    yref_start = 0
    A = 2
    B = 2
    a = 2*np.pi/50
    b = 3*a
    T = np.round(2*np.pi/(a*time_step))
    k = k % T
    delta = np.pi/2
    xref = xref_start + A*np.sin(a*k*time_step + delta)
    yref = yref_start + B*np.sin(b*k*time_step)
    v = [A*a*np.cos(a*k*time_step + delta), B*b*np.cos(b*k*time_step)]
    thetaref = np.arctan2(v[1], v[0])
    return [xref, yref, thetaref]

traj = lissajous
cur_ref = []
for i in range(200):
    cur_ref.append(traj(i))

def check_collision(x,y):
    x_c1,y_c1 = -2,-2
    x_c2, y_c2 = 1,2
    # w_x, w_y = np.random.normal(0,0.04,2)
    w_x,w_y = 0,0
    k = 25 #prev value : 25
    if (x + w_x - x_c1)**2 + (y + w_y  - y_c1)**2 < 0.55**2 :
        penalty = k
    elif (x + w_x - x_c2)**2 + (y + w_y  - y_c2)**2 < 0.55**2 :
        penalty = k 
    else: 
        penalty = 0
    return penalty



def step_cost(e, u, t, ref = cur_ref):
    '''
    : given the current error and the control
    : compute the step cost defined as the sum of tracking errors and control effort
    '''
    Q = 75*np.eye(2)  #Prev :Q= 15, q = 5 for u : 5 X 10; Q = 50, q = 15 for u = 10 X 10
    q = 30
    R = 1*np.eye(2)
    p = e[:2]
    th = e[-1]
    u = np.array([[u[0]],[u[1]]])

    penalty = check_collision(p[0] + ref[t][0], p[1] + ref[t][1])
    return p.T @ Q @ p + q*(1 - np.cos(th))**2 + u.T @ R @ u + penalty

L = np.zeros((n_states, n_controlspace))

@threadpool
def create_L(i):
    e = X[i][1:]
    t = int(X[i][0]//0.5)
    l = np.zeros(n_controlspace)
    for j in range(n_controlspace):
        l[j] = step_cost(np.array(e), U[j],t)
    L[i] = l


generate = True

print('------------ Creating stage cost matrix -----------')
async def main():
    await asyncio.gather(*[create_L(i) for i in tqdm(range(n_states))])

if generate :

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

else:
    with open('step_cost.pkl', 'rb') as f:
        L = pickle.load(f)[0]

def value_iteration_matrix(V,L,p, thresh = 1e-3):
    gamma = 0.95
    for i in tqdm(range(500000)):
        Q = L + 0.95 * (p @ V[i][:,None])[:,:,-1]
        V[i + 1, :] = np.min(Q, axis = 1)
        pi = np.argmin(Q, axis = 1)
        diff = np.linalg.norm(V[i+1] - V[i], np.inf)
        if i%50 == 0:
            print(f'diff : {diff}')
        if diff < thresh : 
            return pi
    return pi

def policy_iteration(V,L,p,pi, thresh = 1e-3):
    gamma = 0.95

    #policy eval
    for i in tqdm(range(500)):
        v_temp = V[i]
        for k in range(3):
            v_temp = L[:,pi[i]] + gamma * P[:, pi[i]] @ v_temp
        Q = L + gamma * (p @ v_temp)
        V[i+1, :] = np.min(Q, axis = 1)
        pi[i+1,:] = np.argmin(Q,axis=1)
        if np.array_equal(V[i], V[i+1]):
            return pi[i+1]
    return pi[i+1] 

n_states = P.shape[0]
num_iters = 5000
n_controlspace = P.shape[1]
V = np.zeros((num_iters + 1, n_states))

# @threadpool
# def assign_V(i, V,ref = cur_ref):
#     p_x =X[i][1]
#     p_y = X[i][2]
#     t = int(X[i][0]//0.5)
#     penalty = check_collision(p_x + ref[t][0], p_y + ref[t][1])
#     if penalty != 0:
#         V[0][i] = 1000
    
# async def main():
#     await asyncio.gather(*[assign_V(i,V) for i in tqdm(range(n_states))])

# print(f'------ Initialising V ---------')
# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())

Q = 75  #Prev :Q= 15, q = 5 for u : 5 X 10; Q = 50, q = 15 for u = 10 X 10
q = 30
R = 1

pi = np.zeros((num_iters + 1, n_states)).astype(np.uint8)
# pi_opt1 = value_iteration_matrix(V,L,p= P)
pi_opt2 = value_iteration_matrix(V,L,P,1e-9)
# pi_opt2 = policy_iteration(V,L,P,pi)
# rand = np.where(pi_opt1 != pi_opt2)
print(f'optimal policy : {pi_opt2}')

with open(f'pi_{Q}_{q}_{R}_obs=0.55.pkl', 'wb') as f:
    x = [pi_opt2, X, e_x, e_y, th, state_table, U]
    pickle.dump(x,f)

with open(f'step_cost_{Q}_{q}_{R}.pkl', 'wb') as f:
    x = [L]
    pickle.dump(x,f)