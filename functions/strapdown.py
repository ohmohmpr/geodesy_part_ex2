import numpy as np
from functions.geodetic_tools import * 

def strapdown_algorithm(accx, accy, av, dt, x, y, yaw_init):
    
    yaw_before = yaw_init
    v_before_x = 0
    v_before_y = 0
    p_before_x = x[0]
    p_before_y = y[0]
    yaw_s = np.full_like(av, [[0]])
    v_s_x = np.full_like(av, [[0]])
    v_s_y = np.full_like(av, [[0]])
    v_s_m = np.full_like(av, [[0]])
    
    p_s_x = np.full_like(av, [[0]])
    p_s_y = np.full_like(av, [[0]])
    
    nbr = len(av)
    for i in range(nbr):
    
        yaw_s[i] = yaw_before + av[i] * dt
        
        a_pb_x_b = np.cos(yaw_s[i]) * accx[i] - np.sin(yaw_s[i]) * accy[i]
        a_pb_y_b = np.sin(yaw_s[i]) * accx[i] + np.cos(yaw_s[i]) * accy[i]
        
        v_s_x[i] = v_before_x + a_pb_x_b * dt
        v_s_y[i] = v_before_y + a_pb_y_b * dt
        v_s_m[i] = np.sqrt(v_s_x[i]**2 + v_s_y[i]**2)
        
        p_s_x[i] = p_before_x + v_s_x[i] * dt
        p_s_y[i] = p_before_y + v_s_y[i] * dt
        
        yaw_before = yaw_s[i]
        v_before_x = v_s_x[i]
        v_before_y = v_s_y[i]
        p_before_x = p_s_x[i]
        p_before_y = p_s_y[i]
        
    return p_s_x, p_s_y, yaw_s, v_s_m