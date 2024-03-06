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
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import scipy.io as scio
from scipy import signal
from sklearn.neural_network import MLPClassifier
from scipy.signal import savgol_filter
import math
import pywt
from scipy import stats, signal




def feature_rms(series, window, step):
    """Root Mean Square"""
    windows_strided, indexes = moving_window_stride(series.values, window, step)
    return pd.Series(data=np.sqrt(np.mean(np.square(windows_strided), axis=1)), index=series.index[indexes])

def feature_skew(series, window, step):
    """Skewness"""
    windows_strided, indexes =moving_window_stride(series.values, window, step)
    return pd.Series(data=stats.skew(windows_strided, axis=1), index=series.index[indexes])



def feature_mean(series, window, step):
    """Mean value"""
    windows_strided, indexes =moving_window_stride(series.values, window, step)
    return pd.Series(data=np.mean(windows_strided, axis=1), index=series.index[indexes])


def feature_median(series, window, step):
    """Median value"""
    windows_strided, indexes = moving_window_stride(series.values, window, step)
    return pd.Series(data=np.median(windows_strided, axis=1), index=series.index[indexes])


def feature_mnf(series, window, step):
    """Mean Frequency"""
    windows_strided, indexes = moving_window_stride(series.values, window, step)
    freq, power = signal.periodogram(windows_strided, 5120)
    return pd.Series(data=np.sum(power*freq, axis=1) / np.sum(power, axis=1), index=series.index[indexes])


def moving_window_stride(array, window, step):
    """
    Returns view of strided array for moving window calculation with given window size and step
    :param array: numpy.ndarray - input array
    :param window: int - window size
    :param step: int - step lenght
    :return: strided: numpy.ndarray - view of strided array, index: numpy.ndarray - array of indexes
    """
    stride = array.strides[0]
    win_count = math.floor((len(array) - window + step) / step)
    strided = as_strided(array, shape=(win_count, window), strides=(stride*step, stride))
    index = np.arange(window - 1, window + (win_count-1) * step, step)
    return strided, index


def feature_iav(series, window, step):
    """Integral Absolute Value"""
    windows_strided, indexes = moving_window_stride(series.values, window, step)
    return pd.Series(data=np.sum(np.abs(windows_strided), axis=1), index=series.index[indexes])


def feature_aac(series, window, step):
    """Average Amplitude Change"""
    windows_strided, indexes = moving_window_stride(series.values, window, step)
    return pd.Series(data=np.divide(np.sum(np.abs(np.diff(windows_strided)), axis=1), window),
                     index=series.index[indexes])

def feature_ssc(series, window, step, threshold):
    """Slope Sign Change"""
    windows_strided, indexes = moving_window_stride(series.values, window, step)
    return pd.Series(data=np.apply_along_axis(lambda x: np.sum((np.diff(x[:-1]) * np.diff(x[1:])) <= -threshold),
                                              axis=1, arr=windows_strided), index=series.index[indexes])
def feature_zc(series, window, step, threshold):
    """Zero Crossing"""
    windows_strided, indexes = moving_window_stride(series.values, window, step)
    zc = np.apply_along_axis(lambda x: np.sum(np.diff(x[(x < -threshold) | (x > threshold)] > 0)), axis=1,
                             arr=windows_strided)
    return pd.Series(data=zc, index=series.index[indexes])


def feature_mnf(series, window, step):
    """Mean Frequency"""
    windows_strided, indexes =moving_window_stride(series.values, window, step)
    freq, power = signal.periodogram(windows_strided, 5120)
    return pd.Series(data=np.sum(power*freq, axis=1) / np.sum(power, axis=1), index=series.index[indexes])


def feature_mdf(series, window, step):
    """Median Frequency"""
    windows_strided, indexes =moving_window_stride(series.values, window, step)
    freq, power = signal.periodogram(windows_strided, 2000)
    ttp_half = np.sum(power, axis=1)/2
    mdf = np.zeros(len(windows_strided))
    for w in range(len(power)):
        for s in range(1, len(power) + 1):
            if np.sum(power[w, :s]) > ttp_half[w]:
                mdf[w] = freq[s - 1]
                break
    return pd.Series(data=mdf, index=series.index[indexes])


def feature_mnp(series, window, step):
    """Mean Power"""
    windows_strided, indexes = moving_window_stride(series.values, window, step)
    freq, power = signal.periodogram(windows_strided, 2000)
    return pd.Series(data=np.mean(power, axis=1), index=series.index[indexes])

def emg_filter_highpass(x, order = 4, sRate = 1000., lowcut = 20.):
    """ Forward-backward band-pass filtering (IIR butterworth filter) """
    nyq = 0.5 * sRate
    low = lowcut/nyq

    b, a = signal.butter(order,low, 'highpass')
    return signal.filtfilt(b,a,x)
def emg_filter_lowpass(x,highcut, order = 4, sRate = 1000.):
    """ Forward-backward band-pass filtering (IIR butterworth filter) """
    nyq = 0.5 * sRate
    high =highcut/nyq

    b, a = signal.butter(order,high, 'lowpass')
    return signal.filtfilt(b,a,x)

def emg_filter_bandpass(x,  lowcut ,highcut,order = 4, sRate = 1000.):
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

def emg_filter_highpass(x, order = 4, sRate = 1000., lowcut = 20.):
    """ Forward-backward band-pass filtering (IIR butterworth filter) """
    nyq = 0.5 * sRate
    low = lowcut/nyq

    b, a = signal.butter(order,low, 'highpass')
    return signal.filtfilt(b,a,x)
def emg_filter_lowpass(x,highcut, order = 4, sRate = 1000.):
    """ Forward-backward band-pass filtering (IIR butterworth filter) """
    nyq = 0.5 * sRate
    high =highcut/nyq

    b, a = signal.butter(order,high, 'lowpass')
    return signal.filtfilt(b,a,x)

def emg_filter_bandpass(x,  lowcut ,highcut,order = 4, sRate = 1000.):
    """ Forward-backward band-pass filtering (IIR butterworth filter) """
    nyq = 0.5 * sRate
    wnl= lowcut/nyq
    wnh=highcut/nyq

    b, a = signal.butter(order,[wnl,wnh],'bandpass')
    return signal.filtfilt(b,a,x)

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
