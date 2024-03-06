import time
import numpy as np


import joblib
from  sklearn import preprocessing
import multiprocessing as mlt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys,os
import scipy
from numpy import arange
from scipy.signal import butter, filtfilt, hilbert
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert
import numpy as np
from numpy import array, sign, zeros
import pandas as pd
import scipy.io as scio
from scipy import signal
from sklearn.neural_network import MLPClassifier
import pywt
from scipy.signal import savgol_filter



def mvc_dataprocess(path,j):
    df = pd.read_csv(path, header=4)
    df = df.dropna(axis=0, how='all')
    # 带通
    df_list = df.iloc[:, j]
    datafilban = emg_filter_bandpass(list(df_list), 10, 450)
    # 平滑滤波
    datasa = savgol_filter(datafilban, 51, 5)
    # 整流
    df_abs = abs(df_list)
    # # 低通
    datalist = emg_filter_lowpass(df_abs, 5)
    return datalist



def emg_filter_highpass(x, order = 3, sRate = 2000., lowcut = 20.):
    """ Forward-backward band-pass filtering (IIR butterworth filter) """
    nyq = 0.5 * sRate
    low = lowcut/nyq

    b, a = signal.butter(order,low, 'highpass')
    return signal.filtfilt(b,a,x)
def emg_filter_lowpass(x,highcut, order = 3, sRate = 2000.):
    """ Forward-backward band-pass filtering (IIR butterworth filter) """
    nyq = 0.5 * sRate
    high =highcut/nyq

    b, a = signal.butter(order,high, 'lowpass')
    return signal.filtfilt(b,a,x)

def emg_filter_bandpass(x,  lowcut ,highcut,order = 3, sRate = 2000.):
    """ Forward-backward band-pass filtering (IIR butterworth filter) """
    nyq = 0.5 * sRate
    wnl= lowcut/nyq
    wnh=highcut/nyq

    b, a = signal.butter(order,[wnl,wnh],'bandpass')
    return signal.filtfilt(b,a,x)
def cal_std(list1):
    '计算标准差'
    return np.std(list1)
def arrLen(t):
    'marker点化整'
    return int(1000*t)

def extractSlidingSD(data, LI, LW,baseSD):
    '滑窗与标准差比较'
    activation_arr=np.zeros(len(data))
    Row= len(data)
    nW = int((Row - LW) / LI)  # 滑窗数目LI增量、LW宽度


    for w in range(nW):
        dataWindow = data[w * LI:w * LI + LW]

        if np.mean(dataWindow)>baseSD:
            condition=1
        else:condition=0
        activation_arr[w * LI:w * LI + LW]=condition
    return activation_arr
def extractSlidingAverage(data, LI, LW,baseSD):
    activation_arr=np.zeros(len(data))
    Row= len(data)
    nW = int((Row - LW) / LI)  # 滑窗数目LI增量、LW宽度


    for w in range(nW):
        dataWindow = data[w * LI:w * LI + LW]

        if np.mean(dataWindow)>baseSD:
            condition=1
        else:condition=0
        activation_arr[w * LI:w * LI + LW]=condition
    return activation_arr

def changedata(t):
    'marker点化整'
    return int(t*1000)

def baseMVC(datalist,df_marker):
    '计算基线baseline，MVC，t1，t2,基线的时间段'
    base1 =np.mean(datalist[0:changedata(df_marker.iloc[0] - 0.5)])
    base2=np.mean(datalist[changedata(df_marker.iloc[1] +0.5):changedata(df_marker.iloc[2] - 0.5)])
    base3 = np.mean(datalist[changedata(df_marker.iloc[3] + 0.5):changedata(df_marker.iloc[4] -0.5)])
    baseline=min(base1,base2,base3)
    if baseline==base1:
        t1=0
        t2=df_marker.iloc[0]-2
    elif baseline==base2:
        t1=df_marker.iloc[1] +2
        t2=df_marker.iloc[2]-2
    else:
        t1=df_marker.iloc[3] +2
        t2=df_marker.iloc[4]-2

    MVC1 =np.max(datalist[changedata(df_marker.iloc[0]+0.5):changedata(df_marker.iloc[1] - 0.5)])
    MVC2=np.max(datalist[changedata(df_marker.iloc[2] +0.5):changedata(df_marker.iloc[3] - 0.5)])
    MVC3 = np.max(datalist[changedata(df_marker.iloc[4] + 0.5):changedata(df_marker.iloc[5] - 0.5)])

    if np.max([MVC1, MVC2, MVC3])<(np.mean([MVC1, MVC2, MVC3]))*1.5:
        MVC = np.max([MVC1, MVC2, MVC3])
    else:
        MVC = np.sum([MVC1, MVC2, MVC3])-np.max([MVC1, MVC2, MVC3])-np.min([MVC1, MVC2, MVC3])

    return baseline,MVC,t1,t2

def fillone(list,zerowith):
    "填补很短的未激活时间段"
    low=0
    up=0
    for i in range(len(list)-1):
        if list[i]>list[i+1]:
            low=i
        if list[i]<list[i+1]:
            up=i
            if up-low<zerowith:
                list[low:up+1]=np.ones(len(list[low:up])+1)
                # print(list[low:up])
    return list
def gante(list):
    rearr=[]
    width=0
    left=len(list)
    # up=0
    # low=0
    for i in range(len(list) - 1):
        if list[0]==1:
            left=0
        if list[i] < list[i + 1]:
            up=i
            left= up

        if list[i] > list[i + 1]:
            low=i
            width= low-left
    # rearr.append([width,left])
    return width,left

def t_split(path):
    tdf=pd.read_excel(path,sheet_name='Sheet2',index_col=None,usecols=None)
    t1=tdf.iloc[0]*1000
    t2 = tdf.iloc[1]*1000
    t3 = tdf.iloc[2]*1000
    t4 = tdf.iloc[3]*1000
    t5 = tdf.iloc[4]*1000
    t6 = tdf.iloc[5]*1000
    tarr=[t1,t2,t3,t4,t5,t6]
    return tarr

def findOne(list):
    index=len(list)
    for i in range(len(list)):
        if list[i]==1:
            index=i
            break
    return index

def envelopfunc(x):
    xh = signal.hilbert(x)
    xe1 = np.abs(xh)

    return xe1

def general_equation(first_x,first_y,second_x,second_y):
    # 斜截式 y = kx + b
    A = second_y-first_y
    B = first_x-second_x
    C = second_x * first_y - first_x * second_y
    k = -1 * A / B
    b = -1 * C / B
    return k, b
# 输入信号序列即可(list)
def envelope_extraction(signal):
    s = signal.astype(float)
    q_u = np.zeros(s.shape)
    q_l = np.zeros(s.shape)

    # 在插值值前加上第一个值。这将强制模型对上包络和下包络模型使用相同的起点。
    # Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.
    u_x = [0, ]  # 上包络的x序列
    u_y = [s[0], ]  # 上包络的y序列

    l_x = [0, ]  # 下包络的x序列
    l_y = [s[0], ]  # 下包络的y序列

    # 检测波峰和波谷，并分别标记它们在u_x,u_y,l_x,l_中的位置。
    # Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.

    for k in range(1, len(s) - 1):
        if (sign(s[k] - s[k - 1]) == 1) and (sign(s[k] - s[k + 1]) == 1):
            u_x.append(k)
            u_y.append(s[k])

        if (sign(s[k] - s[k - 1]) == -1) and ((sign(s[k] - s[k + 1])) == -1):
            l_x.append(k)
            l_y.append(s[k])

    u_x.append(len(s) - 1)  # 上包络与原始数据切点x
    u_y.append(s[-1])  # 对应的值

    l_x.append(len(s) - 1)  # 下包络与原始数据切点x
    l_y.append(s[-1])  # 对应的值

    # u_x,l_y是不连续的，以下代码把包络转为和输入数据相同大小的数组[便于后续处理，如滤波]
    upper_envelope_y = np.zeros(len(signal))
    lower_envelope_y = np.zeros(len(signal))

    upper_envelope_y[0] = u_y[0]  # 边界值处理
    upper_envelope_y[-1] = u_y[-1]
    lower_envelope_y[0] = l_y[0]  # 边界值处理
    lower_envelope_y[-1] = l_y[-1]

    # 上包络
    last_idx, next_idx = 0, 0
    k, b = general_equation(u_x[0], u_y[0], u_x[1], u_y[1])  # 初始的k,b
    for e in range(1, len(upper_envelope_y) - 1):

        if e not in u_x:
            v = k * e + b
            upper_envelope_y[e] = v
        else:
            idx = u_x.index(e)
            upper_envelope_y[e] = u_y[idx]
            last_idx = u_x.index(e)
            next_idx = u_x.index(e) + 1
            # 求连续两个点之间的直线方程
            k, b = general_equation(u_x[last_idx], u_y[last_idx], u_x[next_idx], u_y[next_idx])

            # 下包络
    last_idx, next_idx = 0, 0
    k, b = general_equation(l_x[0], l_y[0], l_x[1], l_y[1])  # 初始的k,b
    for e in range(1, len(lower_envelope_y) - 1):

        if e not in l_x:
            v = k * e + b
            lower_envelope_y[e] = v
        else:
            idx = l_x.index(e)
            lower_envelope_y[e] = l_y[idx]
            last_idx = l_x.index(e)
            next_idx = l_x.index(e) + 1
            # 求连续两个切点之间的直线方程
            k, b = general_equation(l_x[last_idx], l_y[last_idx], l_x[next_idx], l_y[next_idx])

            # 也可以使用三次样条进行拟合
    # u_p = interp1d(u_x,u_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
    # l_p = interp1d(l_x,l_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
    # for k in range(0,len(s)):
    #   q_u[k] = u_p(k)
    #   q_l[k] = l_p(k)

    return upper_envelope_y, lower_envelope_y


def cwt(x, fs, totalscal, wavelet='cgau8'):
    if wavelet not in pywt.wavelist():
        print('小波函数名错误')
    else:
        wfc = pywt.central_frequency(wavelet=wavelet)
        a = 2 * wfc * totalscal /(np.arange(totalscal ,0 ,-1))
        period = 1.0 / fs
        [cwtmar, fre] = pywt.cwt(x, a, wavelet, period)
        amp = abs(cwtmar)
        return amp, fre


def dwt(x ,wavelet='db3'):
    cA, cD = pywt.dwt(x, wavelet, mode='symmetric')
    ya = pywt.idwt(cA, None, wavelet, mode='symmetric')
    yd = pywt.idwt(None ,cD, wavelet ,mode='symmetric')
    return ya, yd, cA, cD

