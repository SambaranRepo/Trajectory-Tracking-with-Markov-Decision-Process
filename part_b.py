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

executor = ThreadPoolExecutor(16)
def threadpool(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        return asyncio.wrap_future(executor.submit(f,*args, **kwargs))
    return wrap

t = np.arange(0,51,0.5)
t = t[:-1]
# e_x = [-3,-2] + list(np.linspace(-1,1,10)) + [2,3]
# e_y = e_x
# th = list(np.linspace(-np.pi,-np.pi/30, 3)) + list(np.linspace(-np.pi/60, np.pi/60, 6)) + list(np.linspace(np.pi/30, np.pi,3))
# e_x= np.linspace(-3,3,8)
# e_y = e_x
# # print(f'e_x : {e_x}')
# th = np.linspace(-np.pi, np.pi, 10)
# X = list(it.product(t,e_x,e_y, th))
e_x = [-3,-1,-0.5,-0.2,0.2, 0.5,1,3]
e_y = e_x
th = [-np.pi, -np.pi/2, -np.pi/4, -np.pi/16,-np.pi/64,np.pi/64,np.pi/16,np.pi/4,np.pi/2,np.pi]
X = list(it.product(t,e_x,e_y, th))
n_states = len(X)
print(f'State space : {n_states}')
state_table = {}
for i in range(n_states):
    state_table.update({X[i]:i})
v = np.linspace(0,1,10)
w = np.linspace(-1,1,10)
U = list(it.product(v,w))
n_controlspace = len(U)
# pdf = multivariate_normal(mean=[0,0,0], cov=np.diag([0.04, 0.04, 0.004]))

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
    ref_diff = np.array([[ref[t][0] - ref[t+1][0]], [ref[t][1] - ref[t+1][1]], [(ref[t][2] -  ref[t+1][2] + np.pi) % (2*np.pi) - np.pi]])
#     print(f'e shape : {e.shape}')
    next_error = e[:,None] + G @ np.array([[u[0]], [u[1]]]) + ref_diff
    # next_error[2] = (next_error[2] + np.pi)%(2*np.pi) - np.pi
#     print(f'next error : {next_error}')
    next_x, next_y, next_th = next_error[0][0], next_error[1][0], next_error[2][0]
    # print(f'next errors : {next_x, next_y, next_th}')
    # print(f'{np.abs(next_x - e_x)}')
    # print(f'{np.abs(next_y - e_y)}')
    # print(f'{np.abs(next_th - th)}')
    ind_x = list(np.abs(next_x - e_x).argsort()[:3])
    ind_y = list(np.abs(next_y - e_y).argsort()[:3])
    ind_th = list(np.abs(next_th - th).argsort()[:3]) 
    # print(f'index x  : {ind_x}')
    next_states, pr = [],[]
    mean = [next_x, next_y, next_th]
    # print(f'mean : {mean}')
    # print(e_x[ind_x[0]])
    pdf = multivariate_normal(mean = mean, cov=np.diag([0.04, 0.04, 0.004]))
    for i in range(3):
        next_states.append(state_table[((t + 1)%50, e_x[ind_x[i]], e_y[ind_y[i]], th[ind_th[i]])])
        # print(f'state probability : {pdf.pdf(e_x[ind_x[i]] - next_x, e_y[ind_y[i]] - next_y, th[ind_th[i]] - next_th)}')
        # print(f'errors : {[list(e_x[ind_x[i]] - next_x)[0], list(e_y[ind_y[i]] - next_y)[0], list(th[ind_th[i]] - next_th)[0]]}')
        pr.append(pdf.pdf([(e_x[ind_x[i]]), (e_y[ind_y[i]]), (th[ind_th[i]])]))
    pr = np.array(pr)
    pr /= (np.sum(pr))
    # print(f'prob : {pr}')
    return next_states, pr

def check_collision(x,y):
    x_c1,y_c1 = -2,-2
    x_c2, y_c2 = 1,2
    k = 50
    if (x-x_c1)**2 + (y-y_c1)**2 < 0.55**2:
        penalty = -k * ((x-x_c1)**2 + (y-y_c1)**2 - 0.55**2)
    elif (x-x_c2)**2 + (y-y_c2)**2 < 0.55**2:
        penalty = -k * ((x-x_c2)**2 + (y-y_c2)**2 - 0.55**2)
    else: 
        penalty = 0
    return penalty

def step_cost(e, u, t, ref = cur_ref):
    '''
    : given the current error and the control
    : compute the step cost defined as the sum of tracking errors and control effort
    '''
    Q = 4* np.eye(2)
    q = 3
    R = 2 * np.eye(2)
    p = e[:2]
    th = e[-1]
    u = np.array([[u[0]],[u[1]]])

    penalty = check_collision(p[0] + ref[t][0], p[1] + ref[t][1])
    return p.T @ Q @ p + q*(1 - np.cos(th))**2 + u.T @ R @ u + penalty


# def threadpool(f):
#     def wrapped(*args, **kwargs):
#         return asyncio.get_event_loop().run_in_executor(executor, f, *args, **kwargs)
#     return wrapped

P = [0] * n_states
coords1,coords2, coords3 = [], [], []
data = []


@threadpool
def create_P(i):
    e = np.array(X[i][1:])
    t = int(X[i][0]//0.5)
    for j in range(n_controlspace):
        next_states, pr = error_dynamics(e,U[j],t)
#         print(f'probs : {pr}')
        for k in range(len(next_states)):
            coords1.append(i)
            coords2.append(j)
            coords3.append(next_states[k])
            data.append(pr[k])    

L = np.zeros((n_states, n_controlspace))

@threadpool
def create_L(i):
    e = X[i][1:]
    t = int(X[i][0]//0.5)
    l = np.zeros(n_controlspace)
    for j in range(n_controlspace):
        l[j] = step_cost(np.array(e), U[j],t)
    L[i] = l


async def main():
    await asyncio.gather(*[create_P(i) for i in tqdm(range(n_states))])
    await asyncio.gather(*[create_L(i) for i in tqdm(range(n_states))])

    # await asyncio.gather(*[create_P(i) for i in tqdm(range(10))])
    # await asyncio.gather(*[create_L(i) for i in tqdm(range(10))])

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
print(f'shape of L : {L.shape}')
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
    save = [p,L]
    pickle.dump(save, f)
