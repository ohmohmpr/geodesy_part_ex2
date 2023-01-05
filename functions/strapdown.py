import numpy as np
from functions.geodetic_tools import * 

def strapdown_algorithm(accx, accy, accz, omgx, omgy, omgz, dt, x_initial_value):
    lat, lon, h = x_initial_value
    f_ib_b = vector3(accx, accy, accz) # 3 x 390172
    w_ib_b = vector3(omgx, omgy, omgz) # 3 x 390172

    c_b_i_0 = np.identity(3)
    v_ib_i_0 = 0
    r_ib_i_0 =  ell2xyz(lat, lon, h) #ECEF
    g = Gravity_ECEF( r_ib_i_0 )
    
    I = np.identity(3)

    v_s = np.full_like(f_ib_b, [[0], [0], [0]])
    p_s = np.full_like(f_ib_b, [[0], [0], [0]])
    euler_angle_s = np.full_like(w_ib_b, [[0], [0], [0]])
    
    c_b_i_before = c_b_i_0
    v_ib_i_before = v_ib_i_0
    r_ib_i_before = r_ib_i_0
    
    nbr = f_ib_b.shape[1]
    # for i in range(1):
    for i in range(nbr):
        S_x = vec2skewmat( w_ib_b[:, i] )
        
        c_b_i_next = c_b_i_before @ (I + S_x * dt)
        
        # Step2: Specific_Force Frame transformation
        f_ib_i = (1/2) * (c_b_i_before + c_b_i_next) @ f_ib_b[:, i]
        
        # Step3: Velocity Update
        a_ib_i = f_ib_i + g
        v_ib_i_next = v_ib_i_before + a_ib_i * dt
        
        # Step4: Position Update
        r_ib_i_next = r_ib_i_before + (v_ib_i_before + v_ib_i_next) * (dt/2)
            
        # Step5: Navigation Solution Transformation
        
        # Update value 
        euler_angle_s[:, i] = rad2deg(Rotmat2Euler(c_b_i_next))
        v_s[:, i] = v_ib_i_next
        p_s[:, i] = r_ib_i_next
        
        # Update value of parameters
        c_b_i_before = c_b_i_next
        v_ib_i_before = v_ib_i_next
        r_ib_i_before = r_ib_i_next
        
    return p_s, v_s, euler_angle_s


def strapdown_algorithm_2D(accx, accy, av, dt, x, y):

    acc = vector2(accx, accy) # 2 x 390172
    
    yaw_before = 0
    
    v_before = np.array([[0], [0]])
    p_before = np.array([[x], [y]])
    
    a_s = np.full_like(acc, [[0], [0]])
    v_s = np.full_like(acc, [[0], [0]])
    p_s = np.full_like(acc, [[0], [0]])
        

    nbr = len(av)
    for i in range(nbr):
        yaw_next = yaw_before + av[i] * dt

        rotation_matrix = np.array([[np.cos(yaw_next), -np.sin(yaw_next)],
                                    [np.sin(yaw_next), np.cos(yaw_next)]])

        acc_matrix = acc[:, i][np.newaxis].T
        
        # ax = [cos(yaw)  -sin(yaw)] * ax_b
        # ay = [sin(yaw)  cos(yaw)] * ay_b
        a_p = rotation_matrix @ acc_matrix

        # V_t = V_t-1 + A_t-1 * dt
        v_next = v_before + a_p * dt
        
        # px = cos(yaw) * vx * dt
        # py = sin(yaw) * vy * dt
        p_next = p_before + np.array([[np.cos(yaw_next), 0], [0, np.sin(yaw_next)]]) @ (v_next * dt)
        
        v_before = v_next
        p_before = p_next
        yaw_before = yaw_next
        
        a_s[:, i] = a_p.T
        v_s[:, i] = v_next.T
        p_s[:, i] = p_next.T
        
    return p_s, v_s, a_s