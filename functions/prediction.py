# prediction function for Kalman Filter (2D)

import numpy as np
import matplotlib.pyplot as plt


def vector6(x1,x2,x3,x4,x5,x6):
    return np.array((x1,x2,x3,x4,x5,x6), dtype=float)  


def prediction(xk, S_xkxk, S_wkwk, dt, i):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Function to predict the state vector of the actual epoch k+1 in a Kalman
    # filter. The motion model from Example 2 in the lectures is used.
    #
    # Input:     xk     ... estimated state vector (epoch k)
    #            S_xkxk ... estimated covariance matrix (epoch k)
    #            S_wkwk ... system noise
    #            dt     ... sampling rate
    #
    # Output:    x_bar  ... pedicted state vector (epoch k+1)
    #            Sx_bar ... predicted covariance matrix (epoch k+1)
    #
    # Author:    M.Sc. Erik Heinz, M.Sc. Tomislav Medic
    # Contact:   heinz@igg.uni-bonn.de, medic@igg.uni-bonn.de
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%
    # Predicted state vector
	
    x_k = xk[0]
    y_k = xk[1]
    angle_k = xk[2] # angle
    omega_k = xk[3] # angular velocity z [rad/s]
    v_k = xk[4]
    a_k = xk[5]
    
    wk_phi = np.sqrt(S_wkwk[0][0])
    wk_a = np.sqrt(S_wkwk[1][1])
    
    x_bar = np.array([x_k + np.cos(angle_k) * v_k * dt,
                      y_k + np.sin(angle_k) * v_k * dt,
                      angle_k + omega_k * dt,
                      omega_k + wk_phi * dt,
                      v_k + a_k * dt,
                      a_k + wk_a * dt])

    # Transition matrix
    Tk = np.array([[1, 0, -np.sin(angle_k) * v_k * dt, 0, np.cos(angle_k) * dt, 0],
                   [0, 1,  np.cos(angle_k) * v_k * dt, 0, np.sin(angle_k) * dt, 0],
                   [0, 0, 1, dt, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, dt],
                   [0, 0, 0, 0, 0, 1]])

    # System noise matrix
    Sk = np.array([[0, 0],
                   [0, 0],
                   [0, 0],
                   [dt, 0],
                   [0, 0],
                   [0, dt],])

    # Predicted covariance matrix
    # Sx_bar = Tk * S_xkxk * Tk' + Sk * S_wkwk * Sk'
    Sx_bar = Tk @ S_xkxk @ Tk.T + Sk @ S_wkwk @ Sk.T

    return x_bar, Sx_bar
# %%
