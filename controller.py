import numpy as np
from casadi import *


def casadi_controller(cur_state, ref):
     Q = 10*np.eye(2)
     q = 10
     R = 5 * np.eye(2)
     cur_ref = ref[0]
     error_x = cur_state[0] - cur_ref[0]
     error_y = cur_state[1] - cur_ref[1]
     error_theta = cur_state[2] - cur_ref[2]
     error_theta = (error_theta + np.pi) % (2 * np.pi) - np.pi
     opti = casadi.Opti()
     N = 15
     U = opti.variable(2,N)
     E = opti.variable(3,N+1)
     error = opti.parameter(3,1)
     opti.set_value(error, vcat([error_x, error_y, error_theta]))
     v = U[0,:]
     w = U[1,:]
     obj = 0
     #For loop to set objective function
     for i in range(N):
          obj += (E[:2,i].T @ Q @ E[:2,i] + q * (1 - cos(E[2,i]))**2 + U[:,i].T @ R @ U[:,i])
     obj += E[:,-1].T @ E[:,-1]
     opti.minimize(obj)
     opti.subject_to(opti.bounded(0, U[0,:], 1))
     opti.subject_to(opti.bounded(-1, U[1,:], 1))
     opti.subject_to(E[:,0] == error)
     for i in range(1,N+1):
          opti.subject_to(E[:,i] == E[:,i-1] + vertcat(hcat([0.5 * cos(E[2, i-1] + ref[i-1][2]), 0]), hcat([0.5 * sin(E[2,i-1] + ref[i-1][2]), 0]), 
          hcat([0, 0.5])) @ U[:,i-1] - vcat([ref[i][0] - ref[i-1][0], ref[i][1] - ref[i-1][1], (ref[i][2] - ref[i-1][2] + np.pi) % (2 * np.pi) - np.pi]))
          
          opti.subject_to(opti.bounded(vcat([-3,-3]), E[:2, i] + vcat([ref[i][0], ref[i][1]]), vcat([3,3])))
          opti.subject_to((E[0,i] + ref[i][0] +  2)**2 + (E[1,i]+ ref[i][1] + 2)**2 > 0.54**2)
          opti.subject_to((E[0,i] + ref[i][0]- 1)**2 + (E[1,i] + ref[i][1] - 2)**2 > 0.54**2)
     
     
     opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
     opti.solver('ipopt', opts)
     sol = opti.solve()
     return sol.value(U)[:,0]


#Add the freespace constraint for the position 
#Implement exact error dynamics by passing the reference trajectory
#How to generalise for time horizon T > 3