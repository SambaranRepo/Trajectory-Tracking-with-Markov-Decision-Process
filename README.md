# Trajectory Tracking

## Overview
In this assignment, we implement a CEC controller and Generalized Policy Iteration Controller to solve a trajectory tracking problem. 

The CEC controller is solved using the CasADi solver which is a symbolic Non Linear Programming solver. 
The generalized policy iteration is implemented as a Value Iteration algorithm.

## Environment Setup

1. Create a conda environment using the ece276b.yaml file provided. 
<pre>conda env create -f ece276b.yaml </pre>
2. Activate the conda environment. 
<pre>conda activate ece276b </pre>

### File description

#### 1. state_control_space.py
This script creates the discretized state space and the control space. It returns the x,y and angle grid as well as a hash table containing the id of different states. Usage : 
<pre>$python3 state_control_space.py</pre> 

#### 2. create_P.py
This script is run once to create the probability transition matrix for a given discretized state space and control space. This matrix is stored in disk for use as offline planner to be used during Value Iteration algorithm. The matrix is saved as a sparse matrix because the size is huge and saving as a numpy float array will not be possible. Usage :
<pre>$python3 create_P.py</pre>

#### 3. VI.py

This script runs the value iteration algorithm on the infinite horizon tracking problem and obtains the optimal policy function. We need to pass the step cost parameters Q, q, R using the terminal and this script will create the policy function for these parameters. Usage : 
<pre>$python3 VI.py 75 30 1</pre> #First parameter is Q, second q and last R

#### 4. casadi_controller.py
This script contains the CEC controller implementation that is solved using the CasADi solver library. The script returns the next control sequence to move the robot, and then the controller is called again from the next time step. We do not need to run the script individually but it is called from the main.py file.

#### 3. main.py

This script calls our custom controller (either CEC or Value Iteration obtained policy controller) and runs the simulation of the robot trying to follow the reference trajectory. We need to pass as argument CEC to call the CEC controller or GPI to call the GPI controller. Usage : 
<pre>$python3 main.py CEC</pre> #Call  the CEC controller
<pre>$python3 main.py GPI</pre> #Call the GPI / Value Iteration obtained optimal policy controller

### Directory 
<pre>
├── fig
├── plots
├── policy
├── report
└── step_cost
├── utils.py
└── VI.py
├── main.py
├── casadi_controller.py
├── create_P.py
</pre>

Results of the trajectory tracking are present in the fig/ and plots/ folders. fig/ contains the gif of the simulation and plots/ contains the plots of the simulation. 

## Technical report
* [Sambaran Ghosal. "Trajectory Tracking" June 2022](report/ECE276B_Trajectory_Tracking.pdf)

## Results 
### Trajectory tracking using the CEC Controller 
<p align='center'>
<img src="fig/animation_CEC_Q7_q5_R2_obs_0.55.gif>
</p>

### Trajectory tracking using optimal policy obtained using Value Iteration Algorithm
<p align='center'>
<img src="fig/animation_GPI_Q75_q30_R1_obs=0.55.gif>
</p>