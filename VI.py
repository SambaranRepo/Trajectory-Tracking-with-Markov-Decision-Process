import numpy as np
import pickle
from tqdm import tqdm
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, thread
import asyncio
import itertools as it

with open('partial_working_MDP.pkl', 'rb') as f:
    x = pickle.load(f)
    L = x[1]
    P = x[0]

executor = ThreadPoolExecutor(16)
def threadpool(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        return asyncio.wrap_future(executor.submit(f,*args, **kwargs))
    return wrap

t = np.arange(0,51,0.5)
t = t[:-1]
e_x= [-3, -0.5] + list(np.linspace(-0.25, 0.25, 5)) + [0.5,3]
e_y = e_x
# print(f'e_x : {e_x}')
th = [-np.pi, -np.pi/2] + list(np.linspace(-np.pi/4, np.pi/4, 5)) + [np.pi/2, np.pi] 
X = list(it.product(t,e_x,e_y, th))
# e_x = [-3,-1,-0.5,-0.2,0.2, 0.5,1,3]
# e_y = e_x
# th = [-np.pi, -np.pi/2, -np.pi/4, -np.pi/16,-np.pi/64,np.pi/64,np.pi/16,np.pi/4,np.pi/2,np.pi]
# X = list(it.product(t,e_x,e_y, th))
n_states = len(X)
print(f'State space : {n_states}')
state_table = {}
for i in range(n_states):
    state_table.update({X[i]:i})
v = np.linspace(0,1,5)
w = np.linspace(-1,1,10)
U = list(it.product(v,w))
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
    k = 500
    if (x-x_c1)**2 + (y-y_c1)**2 < 0.55**2:
        penalty = k 
    elif (x-x_c2)**2 + (y-y_c2)**2 < 0.55**2:
        penalty = k 
    else: 
        penalty = 0
    return penalty

def step_cost(e, u, t, ref = cur_ref):
    '''
    : given the current error and the control
    : compute the step cost defined as the sum of tracking errors and control effort
    '''
    Q = 50*np.eye(2)
    q = 40
    R = 3 * np.eye(2)
    p = e[:2]
    th = e[-1]
    u = np.array([[u[0]],[u[1]]])

    penalty = check_collision(p[0] + ref[t][0], p[1] + ref[t][1])
    return p.T @ Q @ p + q*(1 - np.cos(th))**2 + u.T @ R @ u + penalty


@threadpool
def create_L(i):
    e = X[i][1:]
    t = int(X[i][0]//0.5)
    l = np.zeros(n_controlspace)
    for j in range(n_controlspace):
        l[j] = step_cost(np.array(e), U[j],t)
    L[i] = l

print('------------ Creating stage cost matrix -----------')
async def main():
    await asyncio.gather(*[create_L(i) for i in tqdm(range(n_states))])

loop = asyncio.get_event_loop()
loop.run_until_complete(main())

def value_iteration_matrix(V,pi,L,p):
    for i in tqdm(range(5000)):
        # print(f'shape L : {L.shape}')
        # print(f'p shape : {p.shape}')
        # print(f'shape : {(p @ V)[:,:,-1].shape}')
        # print(np.where((p @ V)[:,:,-1] != 0))
        Q = L + 0.97 * (p @ V[i][:,None])[:,:,-1]
        V[i + 1, :] = np.min(Q, axis = 1)
#         print(f'shape V_new : {V_new.shape}')
        pi = np.argmin(Q, axis = 1)
        diff = np.linalg.norm(V[i+1] - V[i])
        print(f'diff : {diff}')
        if diff < 1e-1 : 
            return pi
    return pi

n_states = P.shape[0]
num_iters = 5000
n_controlspace = P.shape[1]
V = np.zeros((num_iters + 1, n_states))
pi = np.zeros((n_states,2))
pi_opt = value_iteration_matrix(V,pi,L,p= P)
print(f'optimal policy : {pi_opt}')

with open('pi.pkl', 'wb') as f:
    x = [pi_opt]
    pickle.dump(x,f)