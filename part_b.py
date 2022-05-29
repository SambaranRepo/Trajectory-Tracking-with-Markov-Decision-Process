from distutils.log import error
import numpy as np 
import itertools as it
# from scipy.stats import multivariate_normal
from tqdm import tqdm
from scipy.stats import multivariate_normal
# from scipy import sparse
from scipy.sparse import csr_matrix
import asyncio
from sparse import COO
import pickle
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, thread
from state_control_space import *

executor = ThreadPoolExecutor()
def threadpool(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        return asyncio.wrap_future(executor.submit(f,*args, **kwargs))
    return wrap

X,e_x,e_y,th,state_table = generate_state_space()
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


def error_dynamics(e, u,t, ref = cur_ref):
    '''
    : given current error, time, control, current reference and next reference
    : output -> Next error
    : Consider noise in the motion model. 
    '''
    G = np.array([[0.5 * np.cos(e[2] + ref[t][2]), 0], [0.5 * np.sin(e[2] + ref[t][2]), 0], [0, 0.5]])
    ref_diff = np.array([[ref[t][0] - ref[t+1][0]], [ref[t][1] - ref[t+1][1]], [(ref[t][2] -  ref[t+1][2])]])
    next_error = e[:,None] + G @ np.array([[u[0]], [u[1]]]) + ref_diff
    next_error[2] = (next_error[2] + np.pi)%(2*np.pi) - np.pi
    next_x, next_y, next_th = next_error[0][0], next_error[1][0], next_error[2][0]
    k = 4
    ind_x = list(np.abs(next_x - e_x).argsort()[:k])
    ind_y = list(np.abs(next_y - e_y).argsort()[:k])
    ind_th = list(np.abs(next_th - th).argsort()[:k]) 
    next_states, pr = [],[]
    mean = [next_x, next_y, next_th]
    pdf = multivariate_normal(mean = mean, cov=np.diag([0.04, 0.04, 0.004]))
    for i in range(k):
        next_states.append(state_table[(0.5*(t + 1)%50, e_x[ind_x[i]], e_y[ind_y[i]], th[ind_th[i]])])
        pr.append(pdf.pdf([(e_x[ind_x[i]]), (e_y[ind_y[i]]), (th[ind_th[i]])]))
    pr = np.array(pr)
    pr /= (np.sum(pr))
    return next_states, pr


coords1,coords2, coords3 = [], [], []
data = []


@threadpool
def create_P(i):
    e = np.array(X[i][1:])
    t = int(X[i][0]//0.5)
    for j in range(n_controlspace):
        next_states, pr = error_dynamics(e,U[j],t)
        for k in range(len(next_states)):
            coords1.append(i)
            coords2.append(j)
            coords3.append(next_states[k])
            data.append(pr[k])    




async def main():
    await asyncio.gather(*[create_P(i) for i in tqdm(range(n_states))])
    # await asyncio.gather(*[create_P(i) for i in tqdm(range(10))])
    # await asyncio.gather(*[create_L(i) for i in tqdm(range(10))])

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
# print(f'shape of L : {L.shape}')
print(f'coords1 shape : {len(coords1)}')
print(f'coords2 shape : {len(coords2)}')
print(f'coords1 shape : {len(coords3)}')
print(f'data shape : {len(data)}')
coords = np.array([coords1, coords2, coords3])
print(f'coords shape : {coords.shape}')
p = COO(coords, data, shape = (n_states, n_controlspace,n_states))
print(f' shape of p : {p.shape}')

# with open('MDP.pkl', 'rb') as f:
#     p = pickle.load(f)[0]

with open('MDP.pkl', 'wb') as f:
    save = [p]
    pickle.dump(save, f)
