import numpy as np
import pickle
from tqdm import tqdm

with open('MDP.pkl', 'rb') as f:
    x = pickle.load(f)
    L = x[1]
    P = x[0]

print(f'Probability : {np.sum(P[500][20].todense())}')

def value_iteration_matrix(V,pi,L,p):
    for i in tqdm(range(200)):
        # print(f'shape L : {L.shape}')
        # print(f'p shape : {p.shape}')
        # print(f'shape : {(p @ V)[:,:,-1].shape}')
        # print(np.where((p @ V)[:,:,-1] != 0))
        Q = L + 0.9 * (p @ V[i][:,None])[:,:,-1]
        V[i + 1, :] = np.min(Q, axis = 1)
#         print(f'shape V_new : {V_new.shape}')
        pi = np.argmin(Q, axis = 1)
        diff = np.linalg.norm(V[i+1] - V[i])
        print(f'diff : {diff}')
        if diff < 1e-2 : 
            return pi
    return pi

n_states = P.shape[0]
num_iters = 200
n_controlspace = P.shape[1]
V = np.zeros((num_iters + 1, n_states))
pi = np.zeros((n_states,2))
pi_opt = value_iteration_matrix(V,pi,L,p= P)
print(f'optimal policy : {pi_opt}')

with open('pi.pkl', 'wb') as f:
    x = [pi_opt]
    pickle.dump(x,f)