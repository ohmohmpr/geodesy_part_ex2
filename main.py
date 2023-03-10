import numpy as np
import scipy.io
from scipy import interpolate
from numpy.linalg import inv
import matplotlib.pyplot as plt


from functions.geodetic_tools import * 
from functions.prediction import prediction
from functions.timesync import timesync
from functions.readIMARdata import *
from functions.strapdown import strapdown_algorithm


## Mobile Sensing and Robotics - Exercise 2 - IGG Bonn, 12.10.20
# Author 1: M.Sc. Erik Heinz,
# Author 2: M.Sc. Tomislav Medic, 
# Author 3: B.Sc. Felix Esser (Python Code)
# Contact: medic@igg.uni-bonn.de, Tomislav Medic
#          s7feesse@uni-bonn.de, Felix Esser (Python)

# -----------------------------------------------------
# read data
imar_data = IMARdata()
imar_data.readIMAR("./data/IMAR.mat", correctedIMUdata = True)

# -----------------------------------------------------
# Measurements

# Info IMU
f = 1000                  # Measuring frequency [Hz]
dt = 1/f                  # Measuring rate [s]

# Measurements
x = imar_data.gpsUTM[:,0]                 # East UTM [m]
y = imar_data.gpsUTM[:,1]                 # North UTM [m]
a = -imar_data.acceleration[:,0]          # acceleration x [m/s^2]
omega = imar_data.angularvelocity[:,2]    # angular velocity z [rad/s]

# -----------------------------------------------------
# Strapdown algorithm

accx = -imar_data.acceleration[:,0]
accy = -imar_data.acceleration[:,1]
yaw_init = imar_data.rpy_ned[:, 2][0]
x_strapdown, y_strapdown, yaw_s, v_s_m = strapdown_algorithm(accx, accy, omega, dt, x, y, yaw_init)

# -----------------------------------------------------
# Time Synchronisation
idx_gps, idx_imu = timesync( imar_data.gpstime, imar_data.imutime, 0.0005 )

# -----------------------------------------------------
# GPS Error Simulation

# (1) Simulation with normal distrubution
GPS_err_simu = False

if (GPS_err_simu == True):

    # index noize increase
    bd_idx = np.arange(start=700, stop=750, step=1, dtype=int)

    # parameter for normal distribution
    mu_e, sigma_e = 0, 1
    mu_n, sigma_n = 0, 1

    # values from normal distribution
    east_err = np.random.normal( mu_e, mu_e, len(bd_idx) )
    north_err = np.random.normal( mu_n, sigma_n, len(bd_idx) )

    # add noize to GPS observations
    x[bd_idx] += east_err
    y[bd_idx] += north_err

# (2) Multipath Simulation (systematic error)
GPS_multi_simu = False 

if (GPS_multi_simu == True):

    # length of GPS breakdown index
    bd_idx = np.arange(start=1200, stop=1250, step=1, dtype=int)

    # add multipath systematic to GPS observations 
    x[bd_idx] += 2
    y[bd_idx] += -1.3

# TODO (3) choose noise model for GPS

# --------------------------------------------------- #
# ########### Extended Kalman Filter ################ #
# --------------------------------------------------- #

# -----------------------------------------------------
# Statistics

# Measurement noise

std_gps = 0.01     # GPS [m]                               default = 0.05 18.668
std_a = np.std(a[0:100000])      # Accelerations [m/s^2]    default = np.std(a[0:1000])
std_omega = 0.25   # Angular rate [rad/s] default = np.std(omega[0:1000])
# print(std_a)       
# print(std_omega) 

# 1 10000 0.011497345961894776  0.0007721008383805309 7.6758095263681
# 2 100000 0.14706149234759655  0.17653376045083508 4.014702719624097
# 4 sample2 0.14706149234759655 0.2 3.8893015069379486

# 6 sample3 0.14706149234759655 0.25 3.5060247557832924

# 7 sample4 0.14706149234759655 0.26  3.5343471900378876
# 5 sample1 0.14706149234759655 0.3 4.89484
# 8 sample1 0.14706149234759655 0.4 4.9448412809914615
# 3 all 0.1865740396835424  0.1241192696654598 4.203532436537764
# System noise
wk_phi = 0.1       # Angular accelerations [rad/s] default = 0.1  overestimate 10000
wk_a = 1.5         # Linear jerk [m/s^3]          default = 1.5  overestimate 10000

# Covariance Matrix Measurements
Sll = np.array([[std_gps**2, 0, 0, 0],
                [0, std_gps**2, 0, 0],
                [0, 0, std_a**2, 0],
                [0, 0, 0, std_omega**2]])

# Covariance Matrix System Noise
S_wkwk = np.array([[wk_phi**2, 0],
                  [0, wk_a**2]])

std_x_gps = std_gps # RTK GPS
std_y_gps = std_gps # RTK GPS
std_angle = 0.05 # IMU
std_av = std_omega    # IMU
std_v = 0.05     # IMU
std_acc = std_a   # IMU
# Assume
# Covariance Matrix Initial States
S_xkxk = np.array([[std_x_gps**2, 0, 0, 0, 0, 0],
                  [0, std_y_gps**2, 0, 0, 0, 0],
                  [0, 0, std_angle**2, 0, 0, 0],
                  [0, 0, 0, std_av**2, 0, 0],
                  [0, 0, 0, 0, std_v**2, 0],
                  [0, 0, 0, 0, 0, std_acc**2]])
# -----------------------------------------------------


# -----------------------------------------------------
# Measurements

# define zero block
zero_block = np.nan * np.zeros( (2,len(imar_data.imutime) ) )

# Measurement matrix with IMU measurements
L = np.vstack(( np.transpose(imar_data.imutime) , zero_block, np.transpose(a), np.transpose(omega) ))

# add GPS at intersection time stamps
L[1:3,idx_imu] = np.vstack(( np.transpose( x[idx_gps] ), np.transpose( y[idx_gps] ))) 

# -----------------------------------------------------
# Initial states
x_k = x[0]
y_k = y[0]
angle_k = 0 # yaw -0.785398 yaw_init = -0.8964753746986389
omega_k = 0
v_k = 0
a_k = 0 

xk = np.array([x_k, y_k, angle_k, omega_k, v_k, a_k])

# -----------------------------------------------------
# Variables to save estimated states

# states
xstate = np.zeros((len(imar_data.imutime),7)) # (390172, 7)
xstate[0,0] = imar_data.imutime[0] # xstate[0, 0] = time
xstate[0,1:7] = xk

# -------------------------------------------

# Design matrix (Jacobian)
H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1],
              [0, 0, 0, 1, 0, 0]])

# -------------------------------------------
# loop info
percent = 0.05
nbr = len(imar_data.imutime) # all = len(imar_data.imutime)
print('filter started ... ')

# -------------------------------------------
# MAIN KF Loop

for i in range(0, nbr ):
    # -------------------------------------------
    # percent update 
    if (i / nbr) > percent:
        print('%1.0f' % (percent * 100), '% / 100%' )
        percent += 0.05

    # -------------------------------------------
    # Prediction Step
    x_bar, Sx_bar = prediction( xk, S_xkxk, S_wkwk, dt, i )
    xstate[i,0] = imar_data.imutime[i]
    xstate[i,1:7] = x_bar
    # -------------------------------------------
    # Update Step (IMU only)

    if ( np.isnan( L[1,i]) == True ):

        h_gps = H[2:4, :]
        Sll_gps = Sll[2:, 2:]
        K = Sx_bar @ h_gps.T @ np.linalg.inv(h_gps @ Sx_bar @ h_gps.T + Sll_gps)

        # Update state vector
        x_dach = x_bar + K @ (L[3:,i] - h_gps @ x_bar)

        # Covariance matrix states 
        Sx_dach = (np.identity(6) - K @ h_gps) @ Sx_bar

        # save current estimate
        xstate[i,0] = imar_data.imutime[i]
        xstate[i,1:7] = x_dach
        
        # update states for next iteration
        xk = x_dach
        S_xkxk = Sx_dach
    
    # -------------------------------------------
    # Update (GPS + IMU)
    else:
        
        # Kalman Gain Matrix
        K = Sx_bar @ H.T @ np.linalg.inv(H @ Sx_bar @ H.T + Sll)

        # Update state vector
        x_dach = x_bar + K @ (L[1:,i] - H @ x_bar)

        # Covariance matrix states 
        Sx_dach = (np.identity(6) - K @ H) @ Sx_bar

        # save estimation
        xstate[i,0] = imar_data.imutime[i]
        xstate[i,1:7] = x_dach

        # update states for next iteration
        xk = x_dach
        S_xkxk = Sx_dach

# end MAIN loop
# -------------------------------------------

print( 100, '%')
print('... done')

# --------------------------------------------------- #
# ########### Plot EKF results ################ #
# --------------------------------------------------- #

# -------------------------------------------
# reduce plot by indexing
idx_plot = np.arange(start=0, stop=nbr, step=10, dtype=int)

# -------------------------------------------
# Trajectory plot

# Task 3a: Trajectory - Overestimate Measurement noise
# offset_x = 364870
# offset_y = 5621132
# plt.plot(x - offset_x, y - offset_y, '.b', markersize=12)
# plt.plot(xstate[idx_plot,1] - offset_x, xstate[idx_plot,2] - offset_y, '.r')
# plt.axis('equal')
# plt.title('Overestimate Measurement noise, std = 1', fontsize=14, fontweight='bold')
# plt.xlabel('Easting [m]', fontsize=12, fontweight='bold')
# plt.ylabel('Northing [m]', fontsize=12, fontweight='bold')
# plt.legend(['GPS Measurements', 'EKF Trajectory'])
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# plt.savefig("Trajectory - Overestimate Measurement noise = 1.png")
# plt.show()

# -------------------------------------------
# Trajectory plot

# Task 3a: Trajectory - Underestimate Measurement noise
# 7.431367435240396
# offset_x = 364870
# offset_y = 5621132
# plt.plot(x - offset_x, y - offset_y, '.b', markersize=12)
# plt.plot(xstate[idx_plot,1] - offset_x, xstate[idx_plot,2] - offset_y, '.r')
# plt.axis('equal')
# plt.title('Underestimate Measurement noise, std = 0.0000001', fontsize=14, fontweight='bold')
# plt.xlabel('Easting [m]', fontsize=12, fontweight='bold')
# plt.ylabel('Northing [m]', fontsize=12, fontweight='bold')
# plt.legend(['GPS Measurements', 'EKF Trajectory'])
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# plt.savefig("Trajectory - Underestimate Measurement noise = 0.0000001.png")
# plt.show()

# -------------------------------------------
# Trajectory plot

# Task 3a: Trajectory - Overestimate System noise
# 5.011167772367105
# offset_x = 364870
# offset_y = 5621132
# plt.plot(x - offset_x, y - offset_y, '.b', markersize=12)
# plt.plot(xstate[idx_plot,1] - offset_x, xstate[idx_plot,2] - offset_y, '.r')
# plt.axis('equal')
# plt.title('Overestimate System noise, std = 1000000', fontsize=14, fontweight='bold')
# plt.xlabel('Easting [m]', fontsize=12, fontweight='bold')
# plt.ylabel('Northing [m]', fontsize=12, fontweight='bold')
# plt.legend(['GPS Systems', 'EKF Trajectory'])
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# plt.savefig("Trajectory - Overestimate System noise = 1000000.png")
# plt.show()

# -------------------------------------------
# Trajectory plot

# Task 3a: Trajectory - Underestimate System noise
# 644.4679547066012
# offset_x = 364870
# offset_y = 5621132
# plt.plot(x - offset_x, y - offset_y, '.b', markersize=12)
# plt.plot(xstate[idx_plot,1] - offset_x, xstate[idx_plot,2] - offset_y, '.r')
# plt.axis('equal')
# plt.title('Underestimate System noise, std = 0.0000001', fontsize=14, fontweight='bold')
# plt.xlabel('Easting [m]', fontsize=12, fontweight='bold')
# plt.ylabel('Northing [m]', fontsize=12, fontweight='bold')
# plt.legend(['GPS Systems', 'EKF Trajectory'])
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# plt.savefig("Trajectory - Underestimate System noise = 0.0000001.png")
# plt.show()

# -------------------------------------------
# Trajectory plot

# Task 3a: Initial value = 0 std =10000
# offset_x = 364870
# offset_y = 5621132
# plt.plot(x - offset_x, y - offset_y, '.b', markersize=12)
# plt.plot(xstate[idx_plot,1] - offset_x, xstate[idx_plot,2] - offset_y, '.r')
# plt.axis('equal')
# plt.title('Trajectory - Initial value = 0, STD = 10000', fontsize=14, fontweight='bold')
# plt.xlabel('Easting [m]', fontsize=12, fontweight='bold')
# plt.ylabel('Northing [m]', fontsize=12, fontweight='bold')
# plt.legend(['GPS Measurements', 'EKF Trajectory'])
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# plt.savefig("Trajectory - Initial value = 0, STD = 10000.png")
# plt.show()

# -------------------------------------------
# Trajectory plot
# Task 3b: Present the best one
# 5.001781388514242

# offset_x = 364870
# offset_y = 5621132
# plt.plot(x - offset_x, y - offset_y, '.b', markersize=12)
# plt.plot(xstate[idx_plot,1] - offset_x, xstate[idx_plot,2] - offset_y, '.r')
# plt.axis('equal')
# plt.title('GPS Measurement and EKF Trajectory', fontsize=14, fontweight='bold')
# plt.xlabel('UTM (East) [m]', fontsize=12, fontweight='bold')
# plt.ylabel('UTM (North) [m]', fontsize=12, fontweight='bold')
# plt.legend(['GPS Measurements', 'EKF Trajectory'])
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# plt.savefig("GPS Measurement and EKF Trajectory")
# plt.show()

# -------------------------------------------
# Trajectory plot
# Task 3c: find euclidean distance between two points

# gps_x_gps = L[1,idx_imu]
# gps_y_gps = L[2,idx_imu]
# kf_x_gps = xstate[idx_imu, 1]
# kf_y_gps = xstate[idx_imu, 2]
# d = np.sqrt(np.sum((gps_x_gps - kf_x_gps)**2) + np.sum ((gps_y_gps - kf_y_gps)**2))
# print("Different between trajectories = ", d)

# -------------------------------------------
# Acceleration & Angular Velocity

# plt.subplot(211)
# plt.plot( imar_data.imutime , a, '.b' )
# plt.plot( xstate[idx_plot,0], xstate[idx_plot,6], '-r' )
# plt.ylabel(" Acceleration [m/s^2] ", fontsize=12, fontweight='bold')
# plt.legend(["Raw IMU Accelerations", "EKF Accelerations "])
# plt.title("IMU Accelerations vs. Filtered Accelerations (EKF) [X-ACC]", fontsize=14, fontweight='bold' )
# plt.grid(color='k', linestyle='-', linewidth=0.5)
    
# plt.subplot(212)
# plt.plot( imar_data.imutime , geod.rad2deg(omega), '.b' )
# plt.plot( xstate[idx_plot,0], geod.rad2deg(xstate[idx_plot,4]), '-r' )

# plt.ylabel(" Angular Velocity [deg/s] ", fontsize=12, fontweight='bold')
# plt.xlabel(" seconds of day [s] ", fontsize=12, fontweight='bold')
# plt.legend(["Raw IMU Angular Velocity", "EKF Angular Velocity "])
# plt.title("IMU Accelerations vs. Filtered Accelerations (EKF) [YAW] ", fontsize=14, fontweight='bold' )
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# plt.show()

# -------------------------------------------
# Trajectory plot
# Task 4a: All trajectories must be compared in 1 figure/plot.

# offset_x = 364870
# offset_y = 5621132
# plt.plot(x - offset_x, y - offset_y, 'b', markersize=12)
# plt.plot(xstate[idx_plot,1] - offset_x, xstate[idx_plot,2] - offset_y, 'r')
# plt.plot(x_strapdown - offset_x, y_strapdown - offset_y, 'g')
# plt.axis('equal')
# plt.title('Trajectory Analysis - Local grid', fontsize=14, fontweight='bold')
# plt.xlabel('Easting [m]', fontsize=12, fontweight='bold')
# plt.ylabel('Northing [m]', fontsize=12, fontweight='bold')
# plt.legend(['GPS Measurements', 'EKF Trajectory', 'Strapdown'])
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# plt.savefig("GPS Measurement, EKF Trajectory and Strapdown")
# plt.show()

# -------------------------------------------
# Trajectory plot
# Task 4b: find maximum difference between two trajectory

# gps_x_gps = L[1,idx_imu]
# gps_y_gps = L[2,idx_imu]
# kf_x_gps = xstate[idx_imu, 1]
# kf_y_gps = xstate[idx_imu, 2]
# print(kf_x_gps[1008])
# d = np.argmax((gps_x_gps - kf_x_gps)**2 + (gps_y_gps - kf_y_gps)**2)
# print("Find maximum difference = ", d)

# offset_x = 364870
# offset_y = 5621132
# plt.plot(x - offset_x, y - offset_y, '.b', markersize=12)
# plt.plot(xstate[idx_plot,1] - offset_x, xstate[idx_plot,2] - offset_y, '.r')
# plt.plot(gps_x_gps[1008] - offset_x, gps_y_gps[1008] - offset_y, '.r')
# plt.plot(kf_x_gps[1008] - offset_x, kf_y_gps[1008] - offset_y, '.b')
# plt.axis('equal')
# plt.title('Find maximum difference - Local grid', fontsize=14, fontweight='bold')
# plt.xlabel('Easting [m]', fontsize=12, fontweight='bold')
# plt.ylabel('Northing [m]', fontsize=12, fontweight='bold')
# plt.legend(['GPS Measurements', 'EKF Trajectory'])
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# plt.savefig("Find maximum difference")
# plt.show()


# -------------------------------------------
# Trajectory plot
# Task 4b: find maximum difference between two trajectory
# GPS vs strapdown

# gps_x_gps = L[1,idx_imu]
# gps_y_gps = L[2,idx_imu]
# strap_x = x_strapdown[idx_imu]
# strap_y = y_strapdown[idx_imu]

# d = np.argmax((gps_x_gps - strap_x)**2 + (gps_y_gps - strap_y)**2)
# distance = np.max(np.sqrt((gps_x_gps - strap_x)**2 + (gps_y_gps - strap_y)**2))
# print("Find idx maximum difference = ", d)
# print("Find maximum difference = ", distance)
# diff = np.sqrt(np.sum((gps_x_gps - strap_x)**2) + np.sum ((gps_y_gps - strap_y)**2))
# print("Different between trajectories = ", diff)

# offset_x = 364870
# offset_y = 5621132
# plt.plot(x - offset_x, y - offset_y, '.b')
# plt.plot(strap_x - offset_x, strap_y - offset_y, '.r')
# plt.plot([gps_x_gps[d] - offset_x, strap_x[d] - offset_x], [gps_y_gps[d] - offset_y, strap_y[d] - offset_y], '-ok', mfc='C1', mec='C1')
# annotate_x = (gps_x_gps[d] + strap_x[d]) / 2 - offset_x
# annotate_y = (gps_y_gps[d] + strap_y[d]) / 2 - offset_y
# plt.annotate("difference = 9.540 [m]",
#                   xy=(annotate_x , annotate_y), xycoords='data',
#                   xytext=(annotate_x + 8, annotate_y + 8), textcoords='data',
#                   size=10, va="center", ha="center",
#                   arrowprops=dict(arrowstyle="-|>",
#                                   connectionstyle="arc3,rad=-0.2",
#                                   fc="w"),
#                   )
# plt.axis('equal')
# plt.title('Maximum difference - GPS vs Strapdown', fontsize=14, fontweight='bold')
# plt.xlabel('Easting [m]', fontsize=12, fontweight='bold')
# plt.ylabel('Northing [m]', fontsize=12, fontweight='bold')
# plt.legend(['GPS Measurements', 'Strapdown'])
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# plt.savefig("Find maximum difference - GPS vs Strapdown")
# plt.show()


# -------------------------------------------
# Trajectory plot
# Task 4b: find maximum difference between two trajectory
# GPS vs EKF

# gps_x_gps = L[1,idx_imu]
# gps_y_gps = L[2,idx_imu]
# kf_x_gps = xstate[idx_imu, 1]
# kf_y_gps = xstate[idx_imu, 2]

# d = np.argmax((gps_x_gps - kf_x_gps)**2 + (gps_y_gps - kf_y_gps)**2)
# distance = np.max(np.sqrt((gps_x_gps - kf_x_gps)**2 + (gps_y_gps - kf_y_gps)**2))
# print("Find idx maximum difference = ", d)
# print("Find maximum difference = ", distance)
# diff = np.sqrt(np.sum((gps_x_gps - kf_x_gps)**2) + np.sum ((gps_y_gps - kf_y_gps)**2))
# print("Different between trajectories = ", diff)

# offset_x = 364870
# offset_y = 5621132
# plt.plot(x - offset_x, y - offset_y, '.b')
# plt.plot(kf_x_gps - offset_x, kf_y_gps - offset_y, '.r')
# plt.plot([gps_x_gps[d] - offset_x, kf_x_gps[d] - offset_x], [gps_y_gps[d] - offset_y, kf_y_gps[d] - offset_y], '-ok', mfc='C1', mec='C1')
# annotate_x = (gps_x_gps[d] + kf_x_gps[d]) / 2 - offset_x
# annotate_y = (gps_y_gps[d] + kf_y_gps[d]) / 2 - offset_y
# plt.annotate("difference = 0.435 [m]",
#                   xy=(annotate_x , annotate_y), xycoords='data',
#                   xytext=(annotate_x + 0.5, annotate_y + 0.5), textcoords='data',
#                   size=10, va="center", ha="center",
#                   arrowprops=dict(arrowstyle="-|>",
#                                   connectionstyle="arc3,rad=0.2",
#                                   fc="w"),
#                   )
# plt.axis('equal')
# plt.title('Maximum difference - GPS vs EKF', fontsize=14, fontweight='bold')
# plt.xlabel('Easting [m]', fontsize=12, fontweight='bold')
# plt.ylabel('Northing [m]', fontsize=12, fontweight='bold')
# plt.legend(['GPS Measurements', 'EKF'])
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# plt.savefig("Find maximum difference - GPS vs EKF")
# plt.show()

# -------------------------------------------
# Trajectory plot
# Task 4c: Difference in position over time
# GPS vs strapdown vs EKF

gps_time = L[0,idx_imu].shape
print(gps_time)
# gps_x_gps = L[1,idx_imu]
# gps_y_gps = L[2,idx_imu]

# total_time = gps_time[-1] - gps_time[0]
# print(total_time)
# strap_x = x_strapdown[idx_imu]
# strap_y = y_strapdown[idx_imu]

kf_x_gps = xstate[:, 1].shape
print(kf_x_gps)
# kf_y_gps = xstate[idx_imu, 2]

# diff_gps_and_strap = np.sqrt((gps_x_gps - strap_x)**2 + (gps_y_gps - strap_y)**2)
# diff_gps_and_kf = np.sqrt((gps_x_gps - kf_x_gps)**2 + (gps_y_gps - kf_y_gps)**2)

# diff_gps_and_strap_total = np.sqrt(np.sum(((gps_x_gps - strap_x)**2) + (gps_y_gps - strap_y)**2))
# diff_gps_and_kf_total = np.sqrt(np.sum(((gps_x_gps - kf_x_gps)**2) + (gps_y_gps - kf_y_gps)**2))

# print("sum time gps and strap", np.sum(diff_gps_and_strap_total)) # 322.48051196216613 [m]
# print("sum time gps and EKF", np.sum(diff_gps_and_kf_total)) #  5.001781388514242 [m]

# print("average difference per time gps and strap [cm/s]", np.sum(diff_gps_and_strap_total)/total_time * 10e-2)
# print("average difference per time gps and EKF [cm/s]", np.sum(diff_gps_and_kf_total)/total_time * 10e-2)

# plt.plot(gps_time, diff_gps_and_strap, '.b')
# plt.plot(gps_time, diff_gps_and_kf, '.r')
# plt.title('Difference in position over time', fontsize=14, fontweight='bold')
# plt.xlabel('seconds of day [s] ', fontsize=12, fontweight='bold')
# plt.ylabel('Difference [m]', fontsize=12, fontweight='bold')

# plt.legend(['GPS vs Strapdown', 'GPS vs EKF'])
# plt.savefig("Difference in position over time")
# plt.show()

# -------------------------------------------
# Trajectory plot
# Task 4c: Analyze the yaw or heading direction.
# Strapdown and EKF


# plt.subplot(211)
# plt.plot( imar_data.imutime[idx_plot] ,geod.rad2deg(yaw_s[idx_plot]), '-b' )
# plt.plot( xstate[idx_plot,0], geod.rad2deg(xstate[idx_plot,3]), '-r' )

# plt.ylabel('Yaw [degrees]', fontsize=12, fontweight='bold')
# # plt.xlabel('seconds of day [s] ', fontsize=12, fontweight='bold')
# plt.legend(['Strapdown', 'EKF'])
# plt.title('Yaw - Strapdown and EKF', fontsize=14, fontweight='bold')
# plt.grid(color='k', linestyle='-', linewidth=0.5)

# plt.subplot(212)
# diff = geod.rad2deg(yaw_s[idx_plot]) - geod.rad2deg(xstate[idx_plot,3])
# plt.plot( imar_data.imutime[idx_plot] , diff, '-g' )

# plt.ylabel('Difference of Yaw [degrees]', fontsize=12, fontweight='bold')
# plt.xlabel('seconds of day [s] ', fontsize=12, fontweight='bold')
# plt.legend(['Difference Strapdown and EKF'])
# plt.title('Difference of Yaw - Strapdown and EKF', fontsize=14, fontweight='bold')
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.savefig("Yaw - Strapdown and EKF")
# plt.show()


# -------------------------------------------
# Trajectory plot
# Task 4c: Analyze velocity
# Strapdown and EKF

# plt.subplot(211)
# plt.plot( imar_data.imutime[idx_plot] , v_s_m[idx_plot], '-b' )
# plt.plot( xstate[idx_plot,0], np.abs(xstate[idx_plot,5]), '-r' )

# plt.ylabel('Velocity [m/s]', fontsize=12, fontweight='bold')
# plt.legend(['Strapdown', 'EKF'])
# plt.title('Velocity - Strapdown and EKF', fontsize=14, fontweight='bold')
# plt.grid(color='k', linestyle='-', linewidth=0.5)

# plt.subplot(212)
# diff = v_s_m[idx_plot] - np.abs(xstate[idx_plot,5])
# plt.plot( imar_data.imutime[idx_plot] , diff, '-g' )

# plt.ylabel('Difference of Velocity [m/s]', fontsize=12, fontweight='bold')
# plt.xlabel('seconds of day [s] ', fontsize=12, fontweight='bold')
# plt.legend(['Difference Strapdown and EKF'])
# plt.title('Difference of Velocity - Strapdown and EKF', fontsize=14, fontweight='bold')
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# plt.show()

# -------------------------------------------