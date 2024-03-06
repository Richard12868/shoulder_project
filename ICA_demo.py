import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import pandas as pd
from emg_func import feature_mean
from scipy import stats, signal
import matplotlib.mlab as mlab


class ICA_EMG:
    def __init__(self, path):
        self.path=path
        # self.relax_path=relax_path
        self.read_func()


    def read_func(self):
        self.df = pd.read_csv(self.path, header=4)
        # P-R间期0.12-0.2秒，QT间期0.36-0.44秒
        # self.ecg=self.df_relax .iloc[6755:6755+int(0.44*2000),5]*1000000
        # plt.figure()
        # plt.plot(self.ecg,'b')
        # plt.show()
        # self.df = pd.read_csv(self.path, header=4).iloc[0:2000, 5] * 1000000
        # for j in range(int(len(self.df)/2000)):
        #     plt.figure()
        #     data=
        #     plt.plot(self.df)
        #     plt.show()

    def mean_func(self):

        func_name='mean'

        ecg_template = eval('feature_' + func_name + '(' + 'self.ecg' + ',10,1)').tolist()
        plt.plot(ecg_template,'r')
        plt.show()
        # max_value=max(ecg_template)
        # max_index=[i for i,value in enumerate(ecg_template) if value==max_value]
        # print('len',len(ecg_template))
        # print(max_index)
        plt.figure()
        emg = eval('feature_' + func_name + '(' + 'self.df' + ',200,100)').tolist()
        plt.plot(emg,'r')
        plt.show()



    def plot_psd(self):
        plt.figure()
        signal=self.emg_filter_lowpass(self.data)
        plt.figure()
        plt.plot(signal)
        plt.figure()
        plt.psd(signal, NFFT=256, Fs=2000, Fc=0, detrend=mlab.detrend_none,
                window=mlab.window_hanning, noverlap=0)
        # plt.title(f"{name}-{w * 5}s-{w * 5 + 5}s")
        plt.show()

    def ica_emg(self):
        # self.ssa(self.data)
        # ecg=self.emg_filter_lowpass(self.data)
        #
        # print(self.data.shape)
        data=self.df.iloc[:,3:6]
        plt.subplot(3,1,1)
        plt.plot(data.iloc[:,0])
        plt.subplot(3,1,2)
        plt.plot(data.iloc[:,1])
        plt.subplot(3,1,3)
        plt.plot(data.iloc[:,2])
        # plt.plot(u[5000:20000,1],'r')
        plt.show()
        ica = FastICA(n_components=2,max_iter=2000,random_state=42)
        # da=[self.data,ecg]
        # data=np.transpose(data)

        u = ica.fit_transform(data)

        print(u.shape)

        # plt.subplot(3,1,1)
        # plt.plot(u[:,0])
        # plt.subplot(3,1,2)
        # plt.plot(u[:,1])
        # plt.subplot(3,1,3)
        # plt.plot(u[:,2])
        # # plt.plot(u)
        # # plt.plot(u[5000:20000,1],'r')
        # plt.show()

    def emg_filter_highpass(self,x, order = 4, sRate = 2000., lowcut = 50):
        """ Forward-backward band-pass filtering (IIR butterworth filter) """
        nyq = 0.5 * sRate
        low = lowcut/nyq

        b, a = signal.butter(order,low, 'highpass')
        return signal.filtfilt(b,a,x)
    def emg_filter_lowpass(self,x, order = 4, sRate = 2000.,highcut=30):
        """ Forward-backward band-pass filtering (IIR butterworth filter) """
        nyq = 0.5 * sRate
        high =highcut/nyq

        b, a = signal.butter(order,high, 'lowpass')
        return signal.filtfilt(b,a,x)
    def ssa(self):
        series=abs(self.df.iloc[:,5])
        series =self.emg_filter_lowpass(series)
        # step1 嵌入
        windowLen = 20  # 嵌入窗口长度
        seriesLen = len(series)  # 序列长度
        K = seriesLen - windowLen + 1
        X = np.zeros((windowLen, K))
        for i in range(K):
            X[:, i] = series[i:i + windowLen]

        # step2: svd分解， U和sigma已经按升序排序
        U, sigma, VT = np.linalg.svd(X, full_matrices=False)

        for i in range(VT.shape[0]):
            VT[i, :] *= sigma[i]
        A = VT

        # 重组
        rec = np.zeros((windowLen, seriesLen))
        for i in range(windowLen):
            for j in range(windowLen - 1):
                for m in range(j + 1):
                    rec[i, j] += A[i, j - m] * U[m, i]
                rec[i, j] /= (j + 1)
            for j in range(windowLen - 1, seriesLen - windowLen + 1):
                for m in range(windowLen):
                    rec[i, j] += A[i, j - m] * U[m, i]
                rec[i, j] /= windowLen
            for j in range(seriesLen - windowLen + 1, seriesLen):
                for m in range(j - seriesLen + windowLen, windowLen):
                    rec[i, j] += A[i, j - m] * U[m, i]
                rec[i, j] /= (seriesLen - j)

        rrr = np.sum(rec, axis=0)  # 选择重构的部分，这里选了全部

        plt.figure()
        for i in range(10):
            plt.figure()
            plt.plot(rec[i, :])
            plt.show()
        #     ax = plt.subplot(5, 2, i + 1)
        #     ax.plot()
        # plt.show()
        #
        # plt.figure(2)
        # plt.plot(series)
        # plt.show()



if __name__ == '__main__':

    path = '/media/shared/liuruida/database/shoulder_emg/2023-12-12-15-17_1-1-1.csv'
    # relax_path='/media/shared/liuruida/database/301/DATA/20230226SHIJI/1.MVC/mvc-relax-02.csv'
    shoulder_emg= ICA_EMG(path)
    # shoulder_emg.ssa()
    # shoulder_emg.plot_psd()
    shoulder_emg.ica_emg()