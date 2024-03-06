import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
from scipy import stats, signal
from scipy import interpolate

import xml.etree.ElementTree as ET

import math

Kp = 2
Ki = 0.1
q0 = 1.0
q1 = 0.0
q2 = 0.0
q3 = 0.0
halfT = 1 / 800
def imu_calcute(gx, gy, gz, ax,ay, az):
    global Kp, Ki, q0, q1, q2, q3


    q0q0 = q0*q0
    q0q1 = q0*q1
    q0q2 = q0*q2
    q1q1 = q1*q1
    q1q3 = q1*q3
    q2q2 = q2*q2
    q2q3 = q2*q3
    q3q3 = q3*q3

    if ax*ay*az==0:
        return 0,0,0

    norm = np.sqrt(ax*ax + ay*ay + az*az);
    ax = ax / norm
    ay = ay / norm
    az = az / norm

    vx = 2*(q1q3 - q0q2)
    vy = 2*(q0q1 + q2q3)
    vz = q0q0 - q1q1 - q2q2 + q3q3

    ex = ay*vz - az*vy
    ey = az*vx - ax*vz
    ez = ax*vy - ay*vx

    exInt=0
    eyInt=0
    ezInt=0
    exInt = exInt + ex * Ki
    eyInt = eyInt + ey * Ki
    ezInt = ezInt + ez * Ki

    gx = gx + Kp*ex + exInt
    gy = gy + Kp*ey + eyInt
    gz = gz + Kp*ez + ezInt


    q0 = q0 + (-q1*gx - q2*gy - q3*gz)*halfT;
    q1 = q1 + (q0*gx + q2*gz - q3*gy)*halfT;
    q2 = q2 + (q0*gy - q1*gz + q3*gx)*halfT;
    q3 = q3 + (q0*gz + q1*gy - q2*gx)*halfT;

    norm = np.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3);
    q0 = q0/norm;
    q1 = q1/norm;
    q2 = q2/norm;
    q3 = q3/norm;

    roll =  math.atan2(2*q2*q3 + 2*q0*q1, -2*q1*q1 - 2*q2*q2 + 1)*57.3
    pitch = math.asin(2*q1*q3 - 2*q0*q2)*57.3
    yaw  =  -math.atan2(2*q1*q2 + 2*q0*q3, -2*q2*q2 -2*q3*q3 + 1)*57.3
    return roll,pitch,yaw

def read_func(path):
    df = pd.read_csv(path,header=2)
    return df
def calculate_position(df):
    roll_arr=[]
    pitch_arr=[]
    yaw_arr=[]
    for i in range(len(df)):
        gx=df.iloc[i,6]
        gy=df.iloc[i,7]
        gz=df.iloc[i,8]
        ax=df.iloc[i,3]*9.8/1000
        ay=df.iloc[i,4]*9.8/1000
        az=df.iloc[i,5]*9.8/1000
        roll,pitch,yaw=imu_calcute(gx, gy, gz, ax,ay, az)
        roll_arr.append(roll)
        pitch_arr.append(pitch)
        yaw_arr.append(yaw)
    plt.plot(roll_arr)
    # plt.plot(yaw_arr)
    plt.show()
if __name__ == '__main__':
    path='/media/shared/liuruida/database/imu-test/2022-11-23-20-11_90X3.csv'
    df=read_func(path)
    calculate_position(df)
