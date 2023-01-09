## main.py ###


######################## YAW CODE ####################################
# yaw_kf = xstate[:, 3]
# av_kf = xstate[:, 4]
# a_kf = xstate[:, 6]
# yaw_rpy_ned = imar_data.rpy_ned[:,2]
# omega = imar_data.angularvelocity[:,2]

# yaw_strapdown = strapdown_algorithm_2D_new(yaw_rpy_ned, a_kf, omega, dt, x, y)
# time_test = imar_data.imutime

# plt.scatter(time_test, yaw_kf)
# plt.scatter(time_test, yaw_strapdown)
# plt.scatter(time_test, yaw_rpy_ned)
# plt.legend(["yaw_kf", "yaw_strapdown", "yaw_rpy_ned"])
# plt.title("yaw_kf vs yaw_strapdown[av correctted y start = -0.81] vs yaw_rpy_ned", fontsize=14, fontweight='bold' )
# plt.show()
######################## YAW CODE ####################################

# ####################### V CODE ####################################
# v_kf = xstate[:, 5]
# a_kf = xstate[:, 6]
# yaw_rpy_ned = imar_data_sd.rpy_ned[:,2]
# accx = -imar_data.acceleration[:,0]
# accy = -imar_data.acceleration[:,1]
# omega = imar_data.angularvelocity[:,2]

# yaw_strapdown, vs_strapdown_m = strapdown_algorithm_2D_new(yaw_rpy_ned, accx, accy, omega, dt, x, y)
# time_test = imar_data_sd.imutime

# plt.scatter(time_test, vs_strapdown_m)
# plt.scatter(time_test, v_kf)
# plt.legend(["vs_strapdown_m[-accx, -accy]", "v_kf"])
# plt.title("vs_strapdown_m[-accx, -accy] vs v_kf", fontsize=14, fontweight='bold' )
# plt.show()
# ####################### V CODE ####################################

####################### V CODE ####################################
# accx = -imar_data.acceleration[:,0]
# accy = -imar_data.acceleration[:,1]
# omega = imar_data.angularvelocity[:,2]

# x_strapdown, y_strapdown, _ , _ = strapdown_algorithm(accx, accy, omega, dt, x, y)
# time_test = imar_data.imutime

# plt.plot(x,y, '.b', markersize=12)
# plt.plot(x_strapdown, y_strapdown, '.g')
# plt.axis('equal')
# plt.title('GPS vs Strapdown', fontsize=14, fontweight='bold')
# plt.xlabel('UTM (East) [m]', fontsize=12, fontweight='bold')
# plt.ylabel('UTM (North) [m]', fontsize=12, fontweight='bold')
# plt.legend(['GPS Measurements', 'Strapdown'])
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# plt.show()
####################### V CODE ####################################










### strapdown.py ######

# def strapdown_algorithm_3D(accx, accy, accz, omgx, omgy, omgz, dt, x_initial_value):
#     '''
#         Find bugs later
#     '''
#     lat, lon, h = x_initial_value
#     f_ib_b = vector3(accx, accy, accz) # 3 x 390172
#     w_ib_b = vector3(omgx, omgy, omgz) # 3 x 390172

#     c_b_i_0 = np.identity(3)
#     v_ib_i_0 = 0
#     r_ib_i_0 =  ell2xyz(lat, lon, h) #ECEF
#     g = Gravity_ECEF( r_ib_i_0 )
    
#     I = np.identity(3)

#     v_s = np.full_like(f_ib_b, [[0], [0], [0]])
#     p_s = np.full_like(f_ib_b, [[0], [0], [0]])
#     euler_angle_s = np.full_like(w_ib_b, [[0], [0], [0]])
    
#     c_b_i_before = c_b_i_0
#     v_ib_i_before = v_ib_i_0
#     r_ib_i_before = r_ib_i_0
    
#     nbr = f_ib_b.shape[1]
#     # for i in range(1):
#     for i in range(nbr):
#         S_x = vec2skewmat( w_ib_b[:, i] )
        
#         c_b_i_next = c_b_i_before @ (I + S_x * dt)
        
#         # Step2: Specific_Force Frame transformation
#         f_ib_i = (1/2) * (c_b_i_before + c_b_i_next) @ f_ib_b[:, i]
        
#         # Step3: Velocity Update
#         a_ib_i = f_ib_i + g
#         v_ib_i_next = v_ib_i_before + a_ib_i * dt
        
#         # Step4: Position Update
#         r_ib_i_next = r_ib_i_before + (v_ib_i_before + v_ib_i_next) * (dt/2)
            
#         # Step5: Navigation Solution Transformation
        
#         # Update value 
#         euler_angle_s[:, i] = rad2deg(Rotmat2Euler(c_b_i_next))
#         v_s[:, i] = v_ib_i_next
#         p_s[:, i] = r_ib_i_next
        
#         # Update value of parameters
#         c_b_i_before = c_b_i_next
#         v_ib_i_before = v_ib_i_next
#         r_ib_i_before = r_ib_i_next
        
#     return p_s, v_s, euler_angle_s


# def strapdown_algorithm_2D(accx, accy, av, dt, x, y):
#     '''
#         Find bugs later
#     '''

#     acc = vector2(accx, accy) # 2 x 390172
    
#     yaw_before = 0
    
#     v_before = np.array([[0], [0]])
#     p_before = np.array([[x], [y]])
    
#     a_s = np.full_like(acc, [[0], [0]])
#     v_s = np.full_like(acc, [[0], [0]])
#     p_s = np.full_like(acc, [[0], [0]])
        

#     nbr = len(av)
#     for i in range(nbr):
#         yaw_next = yaw_before + av[i] * dt

#         rotation_matrix = np.array([[np.cos(yaw_next), -np.sin(yaw_next)],
#                                     [np.sin(yaw_next), np.cos(yaw_next)]])

#         acc_matrix = acc[:, i][np.newaxis].T
        
#         # ax = [cos(yaw)  -sin(yaw)] * ax_b
#         # ay = [sin(yaw)  cos(yaw)] * ay_b
#         a_p = rotation_matrix @ acc_matrix

#         # V_t = V_t-1 + A_t-1 * dt
#         v_next = v_before + a_p * dt
        
#         # px = cos(yaw) * vx * dt
#         # py = sin(yaw) * vy * dt
#         p_next = p_before + np.array([[np.cos(yaw_next), 0], [0, np.sin(yaw_next)]]) @ (v_next * dt)
        
#         v_before = v_next
#         p_before = p_next
#         yaw_before = yaw_next
        
#         a_s[:, i] = a_p.T
#         v_s[:, i] = v_next.T
#         p_s[:, i] = p_next.T
        
#     return p_s, v_s, a_s

