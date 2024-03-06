import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
from scipy import stats, signal
from scipy import interpolate

import xml.etree.ElementTree as ET
import math
import warnings
warnings.filterwarnings("ignore")
#
def emg_filter_highpass(x, order = 4, sRate = 400., lowcut = 20.):
    """ Forward-backward band-pass filtering (IIR butterworth filter) """
    nyq = 0.5 * sRate
    low = lowcut/nyq

    b, a = signal.butter(order,low, 'highpass')
    return signal.filtfilt(b,a,x)
def emg_filter_lowpass(x,highcut, order = 3, sRate = 400.):
    """ Forward-backward band-pass filtering (IIR butterworth filter) """
    nyq = 0.5 * sRate
    high =highcut/nyq

    b, a = signal.butter(order,high, 'lowpass')
    return signal.filtfilt(b,a,x)

def emg_filter_bandpass(x,  lowcut ,highcut,order = 4, sRate = 400.):
    """ Forward-backward band-pass filtering (IIR butterworth filter) """
    nyq = 0.5 * sRate
    wnl= lowcut/nyq
    wnh=highcut/nyq

    b, a = signal.butter(order,[wnl,wnh],'bandpass')
    return signal.filtfilt(b,a,x)
Kp =35
Ki = 0.001
q1 = 1.0
q2 = 0.0
q3 = 0.0
q4 = 0.0
eInt=[0,0,0]
def imu_calcute(gx, gy, gz, ax,ay, az, mx,my, mz,SamplePeriod):
    global Kp, Ki, q1, q2, q3, q4,eInt


    q1q1 = q1*q1
    q1q2 = q1*q2
    q1q3 = q1*q3
    q1q4 = q1*q4
    q2q2 = q2*q2
    q2q3 = q2*q3
    q2q4 = q2*q4
    q3q3 = q3 * q3
    q3q4 = q3*q4
    q4q4 = q4 * q4

    normA = np.sqrt(ax * ax + ay * ay + az * az)
    if normA ==0:
        return 0,0,0
    ax = ax / normA
    ay = ay / normA
    az = az / normA

    normM = np.sqrt(mx * mx + my * my + mz * mz)
    if normM==0:
        return 0,0,0
    mx =mx/ normM
    my =my/normM
    mz =mz/ normM

    hx=2*mx*(0.5-q3q3-q4q4)+2 * my * (q2q3 - q1q4) + 2 * mz * (q2q4 + q1q3)
    hy = 2 * mx * (q2q3 + q1q4) + 2 * my * (0.5 - q2q2 - q4q4) + 2* mz * (q3q4 - q1q2)
    bx =np.sqrt((hx * hx) + (hy * hy))
    bz = 2 * mx * (q2q4 - q1q3) + 2* my * (q3q4 + q1q2) + 2* mz * (0.5- q2q2 - q3q3)
    vx = 2* (q2q4 - q1q3)
    vy = 2* (q1q2 + q3q4)
    vz = q1q1 - q2q2 - q3q3 + q4q4
    wx = 2 * bx * (0.5 - q3q3 - q4q4) + 2* bz * (q2q4 - q1q3)
    wy = 2* bx * (q2q3 - q1q4) + 2* bz * (q1q2 + q3q4)
    wz = 2* bx * (q1q3 + q2q4) + 2* bz * (0.5- q2q2 - q3q3)

    #  Error is cross product between estimated direction and measured direction of gravity
    ex = (ay * vz - az * vy) + (my * wz - mz * wy)
    ey = (az * vx - ax * vz) + (mz * wx - mx * wz)
    ez = (ax * vy - ay * vx) + (mx * wy - my * wx)

    if Ki > 0:
        eInt[0] += ex
        eInt[1] += ey
        eInt[2] += ez

    else:
        eInt[0] = 0.0
        eInt[1] = 0.0
        eInt[2] = 0.0

    # // Apply feedback terms
    gx = gx + Kp * ex + Ki * eInt[0]
    gy = gy + Kp * ey + Ki * eInt[1]
    gz = gz + Kp * ez + Ki * eInt[2]
    # // Integrate  rate  of change of quaternion
    pa = q2
    pb = q3
    pc = q4
    q1 = q1 + (-q2 * gx - q3 * gy - q4 * gz) * (0.5* SamplePeriod)
    q2 = pa + (q1 * gx + pb * gz - pc * gy) * (0.5* SamplePeriod)
    q3 = pb + (q1 * gy - pa * gz + pc * gx) * (0.5* SamplePeriod)
    q4 = pc + (q1 * gz + pa * gy - pb * gx) * (0.5* SamplePeriod)

    # // Normalise quaternion
    norm = np.sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)

    q1 = q1/norm
    q2= q2/ norm
    q3= q3 /norm
    q4 = q4/ norm
    
    roll =  math.atan2(2*q3*q4 + 2*q1*q2, -2*q2*q2 - 2*q3*q3 + 1)*57.3
    pitch = math.asin(2*q2*q4 - 2*q1*q3)*57.3
    yaw  =  -math.atan2(2*q2*q3 + 2*q1*q4, -2*q3*q3 -2*q4*q4 + 1)*57.3
    return roll,pitch,yaw

def read_func(path):
    df = pd.read_csv(path,header=2)
    return df
def calculate_position(df):
    roll_arr=[]
    pitch_arr=[]
    yaw_arr=[]
    sampleperiod=1/400

    for k in range(3,12):
        plt.figure()
        sig=emg_filter_lowpass(df.iloc[:, k],20)
        plt.plot(np.array(df.iloc[:, k]))
        plt.plot(sig, 'r')
        plt.title(df.columns[k])
        plt.show()
    gx_arr = emg_filter_lowpass(df.iloc[:, 6],10)
    gy_arr = emg_filter_lowpass(df.iloc[:, 7],10)
    gz_arr = emg_filter_lowpass(df.iloc[:, 8],10)
    ax_arr =emg_filter_lowpass( df.iloc[:, 3],10) * 9.8 / 1000
    ay_arr = emg_filter_lowpass(df.iloc[:, 4],10) * 9.8 / 1000
    az_arr = emg_filter_lowpass(df.iloc[:, 5],10) * 9.8 / 1000
    mx_arr = emg_filter_lowpass(df.iloc[:, 9],10) / 1000
    my_arr = emg_filter_lowpass(df.iloc[:, 10],10) / 1000
    mz_arr = emg_filter_lowpass(df.iloc[:, 11],10) / 1000



    for i in range(len(df)):
        gx=gx_arr[i]
        gy=gy_arr[i]
        gz=gz_arr[i]
        ax=ax_arr[i]
        ay=ay_arr[i]
        az=az_arr[i]
        mx=mx_arr[i]
        my=my_arr [i]
        mz=mz_arr[i]
        # gx=df.iloc[i, 6]
        # gy=df.iloc[i, 7]
        # gz=df.iloc[i, 8]
        # ax=df.iloc[i, 3] * 9.8 / 1000
        # ay=df.iloc[i, 4] * 9.8 / 1000
        # az=df.iloc[i, 5] * 9.8 / 1000
        # mx=df.iloc[i, 9]  / 1000
        # my=df.iloc[i, 10]  / 1000
        # mz=df.iloc[i, 11] / 1000

        roll,pitch,yaw=imu_calcute(gx, gy, gz, ax,ay, az,mx,my,mz,sampleperiod)
        roll_arr.append(roll)
        pitch_arr.append(pitch)
        yaw_arr.append(yaw)
    # plt.plot(np.array(df.iloc[850:1100, 6]),'b')
    # plt.figure()
    # plt.plot(yaw_arr[750:1100],'r')
    # plt.title('yaw')
    plt.figure()
    plt.plot(emg_filter_lowpass(pitch_arr[10:],5),'r')
    # plt.title('roll')
    # plt.figure()
    # plt.plot(pitch_arr[750:1100],'r')
    # plt.title('pitch')

    plt.show()
if __name__ == '__main__':
    path= '/media/shared/liuruida/database/imu-test/2022-11-23-20-26_90Y3.csv'
    df=read_func(path)
    calculate_position(df)
