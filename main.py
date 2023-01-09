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
x_strapdown, y_strapdown, _ , _ = strapdown_algorithm(accx, accy, omega, dt, x, y, yaw_init)

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

std_gps = 0.01     # GPS [m]                               default = 0.01
std_a = np.std(a[0:1000])       # Accelerations [m/s^2]    default = np.std(a[0:1000])
std_omega = np.std(a[0:1000])   # Angular rate [rad/s] default = np.std(omega[0:1000])

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

gps_x_gps = L[1,idx_imu]
gps_y_gps = L[2,idx_imu]
kf_x_gps = xstate[idx_imu, 1]
kf_y_gps = xstate[idx_imu, 2]
d = np.sqrt(np.sum((gps_x_gps - kf_x_gps)**2) + np.sum ((gps_y_gps - kf_y_gps)**2))
print("Different between trajectories = ", d)

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