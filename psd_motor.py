# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pywt
import argparse
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
def plot_psd(x):

    [y,f]=plt.psd(x, NFFT=256, Fs=2000, Fc=0, detrend=mlab.detrend_none,
            window=mlab.window_hanning, noverlap=0)
    plt.show()
    # print([y,f])
    print(f[np.argmax(y)])

def emg_filter_bandpass(x,  lowcut ,highcut,order = 4, sRate = 2000.):
    """ Forward-backward band-pass filtering (IIR butterworth filter) """
    nyq = 0.5 * sRate
    wnl= lowcut/nyq
    wnh=highcut/nyq

    b, a = signal.butter(order,[wnl,wnh],'bandpass')
    return signal.filtfilt(b,a,x)
def read_csv(path):
    df=pd.read_csv(path,header=3)
    for i in np.arange(3,9):
        list=df.iloc[:,i]

        plot_psd(list)
        plt.figure()
        plt.plot(list)
        plt.show()


if __name__ == '__main__':
    path='/media/shared/liuruida/database/dengsu/2023-07-07-12-09_4-a-120.csv'
    read_csv(path)