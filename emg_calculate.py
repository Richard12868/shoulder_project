# -*- coding: utf-8 -*-


import time
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import sys,os
import scipy
from numpy import arange
from scipy.signal import butter, filtfilt, hilbert
from scipy import signal
from scipy.signal import hilbert
import pandas as pd
import scipy.io as scio
from scipy import signal
from emg_func import *

class Shoulder_EMG:
    def __init__(self, path):
        self.path=path


    def read_func(self):
        self.df = pd.read_csv(self.path, header=4)

    def window_func(self, fname,LW, LI, bool_plot):
        # for i in range(3,12):
        for i in range(8, 12):
            series_data =fname (self.df.iloc[:,i],LW, LI)

            if bool_plot==1:
                # x=np.arange(len(series_data))
                x=series_data.index
                y=series_data
                plt.plot(x,y)
                plt.title(str(fname)+'-'+str(i))
                plt.show()
    def activate_func(self,bool_plot):
        # for i in range(3,12):
        # 带通
        for i in range(3, 13):
            datafilban = emg_filter_bandpass(self.df.iloc[:, i]*1000000, 10, 450)
            # 整流
            df_abs = abs(datafilban)
            # 低通
            datafilow = emg_filter_lowpass(df_abs, 5)

            # 平滑滤波
            dataaver = savgol_filter(datafilow, 1001, 3)
            datalist = abs(dataaver)
            if bool_plot == 1:
                x = np.arange(len(datalist))
                y = datalist
                plt.plot(x, y)
                plt.title(f'activation-{i-2}')
                plt.show()

    def cwt_func(self):
        # for i in range(3, 12):
        for i in range(8, 17):
            # 带通
            datafilban = emg_filter_bandpass(self.df.iloc[:, i], 10, 450)

            fs = 2000

            time = len(self.df)/2000

            t = np.linspace(0, time - 1 / fs, int(time * fs))
            x =datafilban
            amp, fre = cwt(x, fs, 512, 'morl')
            plt.figure(1)
            plt.subplot(2, 1, 1)
            plt.plot(t, x)
            plt.ylabel('Amplitude')
            plt.xlabel('time')
            plt.subplot(2, 1, 2)
            plt.contourf(t, fre, amp)
            plt.ylabel('Frequency')
            plt.xlabel('time')
            # # 离散小波分析
            # ya, yd, _, _ = dwt(x, 'db3')
            # plt.figure(2)
            # plt.plot(t, ya)
            # plt.xlabel('time')
            # plt.ylabel('近似系数')
            plt.show()

    # def window_func(self,festr, LI, LW):
    #
    #         Row, Column = self.data.shape
    #         nW = int((Row - LW) / LI)  # 滑窗数目
    #
    #
    #         feMatrix = np.zeros([nW,  Column])
    #         for w in range(nW):
    #             feRow = np.zeros([1,  Column])
    #             for c in range(Column):
    #                 dataWindow = self.data[w * LI:w * LI + LW, c]
    #
    #                 fe = eval('f' + festr + '(' + str(list(dataWindow)) + ')')
    #                 feRow[0, c] = fe
    #             feMatrix[w, :] = feRow[0, :]
    #         return feMatrix




if __name__ == '__main__':
    fname=feature_aac
    # fname2=feature_mdf

    LW=500
    LI=200
    bool_plot=1
    # 读取目录
    path = '/home/liuruida/Documents/database/shoulder_emg/touzhi-03.csv'

    shoulder_emg= Shoulder_EMG(path)
    shoulder_emg.read_func()
    # shoulder_emg.window_func(feature_rms, LW, LI, bool_plot)
    # shoulder_emg.window_func(feature_aac, LW, LI, bool_plot)
    # shoulder_emg.window_func(feature_mnp,LW, LI, bool_plot)
    # shoulder_emg.window_func(feature_mdf, LW, LI, bool_plot)
    shoulder_emg.activate_func(bool_plot)
    # shoulder_emg.cwt_func()