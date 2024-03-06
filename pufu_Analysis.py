import numpy as np
import pandas as pd
import pywt
import argparse
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import warnings
from emg_func import feature_aac,feature_rms,\
    feature_skew,feature_iav,feature_ssc,feature_zc,feature_mnf,feature_mdf,feature_mnp
warnings.filterwarnings("ignore")
from itertools import product
import os

# ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor']
for family in pywt.families():  #打印出每个小波族的每个小波函数
    print('%s family: '%(family) + ','.join(pywt.wavelist(family)))

class Pufu_analysis:
    def __init__(self,opt):
        self.opt=opt
        self.file_path=opt.data_path
        self.file_name=opt.data_name
        self.filetree_path = opt.filetree_path
        self.dwt_waveletname=opt.dwt_wavelet
        self.savefig_path =opt.savefig_path
        self.fs=opt.fs
        self.cwt_waveletname=opt.cwt_wavelet
        self.subfiletree=opt.subfiletree




    def read_func(self):

        path=self.file_path+'/'+self.file_name
        print(path)
        df = pd.read_csv(path, header=4).iloc[:,3:13]
        print(self.file_path)
        # df = pd.read_csv(path, header=2).iloc[2:]
        self.source_data= df

    def plot_analysis(self,func_name):
        # plt.figure(figsize = (6, 12))
        new_row = pd.Series([None] * 10)

        for i in range(np.shape(self.source_data)[1]):
            # print('feature_'+func_name+'('+'list(self.source_data.iloc[:,i])'+',200,100)')

            data=eval('feature_'+func_name+'('+'self.source_data.iloc[:,i]'+',200,100)')
            feature_mean=np.mean(data)
            # print(i)
            new_row.loc[i] =feature_mean
        return new_row

        #     plt.subplot(10, 1, i+1)
        #     plt.subplots_adjust(hspace=1)
        #     if i==0:
        #         fig_title=func_name+'-'+self.file_name.split('.')[0]
        #         plt.title(fig_title)
        #     plt.plot(data)
        # print('------'+fig_title+'------')
        # plt.savefig(self.savefig_path+'/'+fig_title+'.png')



    # plt.figure()
    # plt.subplot(2, 1, 1)
    def cwt(self,totalscal=100):
        if self.cwt_waveletname not in pywt.wavelist():
            print('小波函数名错误',self.cwt_waveletname)
        else:
            for i in range(np.shape(self.source_data)[1]):
                wfc = pywt.central_frequency(wavelet=self.cwt_waveletname)
                a = 2 * wfc * totalscal /(np.arange(totalscal ,0 ,-1))
                period = 1.0 / self.fs
                [cwtmar, fre] = pywt.cwt(self.source_data.iloc[:,i], a,self.cwt_waveletname, period)
                amp = abs(cwtmar)
                t = np.arange(0, len(self.source_data.iloc[:,i])/2000, 1/2000)
    
                plt.contourf(t, fre[2*len(fre)//3:len(fre)], amp[2*len(fre)//3:len(fre),:])
                plt.ylabel('Frequency')
                plt.xlabel('time')
                plt.title(self.cwt_waveletname)
                plt.show()


    def dwt(self,data ,wavelet='db3'):

        # Create wavelet object and define parameters
        w = pywt.Wavelet('db8')  # 选用Daubechies8小波
        maxlev = pywt.dwt_max_level(len(data), w.dec_len)
        print("maximum level is " + str(maxlev))
        threshold = 0.1 # Threshold for filtering

        # Decompose into wavelet components, to the level selected:
        coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解

        plt.figure()
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波

        datarec = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构

        #
        # plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.plot( data)
        # plt.xlabel('time (s)')
        # plt.ylabel('microvolts (uV)')
        # plt.title("Raw signal")
        # plt.subplot(2, 1, 2)
        # plt.plot(datarec)
        # plt.xlabel('time (s)')
        # plt.ylabel('microvolts (uV)')
        # plt.title("De-noised signal using wavelet techniques")
        # plt.tight_layout()
        # plt.show()
        wave_base=datarec[5350:6350]
        # plt.figure()
        # plt.plot( wave_base)
        # plt.xlabel('time (s)')
        # plt.ylabel('microvolts (uV)')
        # plt.title("Raw signal")
        # plt.show()
        #
        # plt.figure()
        # plt.plot( data[0:1000])
        # plt.xlabel('time (s)')
        # plt.ylabel('microvolts (uV)')
        # plt.title("Raw signal")
        # plt.show()
        return wave_base

    def base_waveFunc(self,data,wavelet,LW=1000,LI=50):


        Row = len(data)
        nW = int((Row - LW) / LI)  # 滑窗数目LI增量、LW宽度
        for w in range(nW):

            dataWindow = data[w * LI:w * LI + LW]
            vec1=wavelet
            vec2=dataWindow
            cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            print(cos_sim)
            # plt.figure()
            # plt.plot(vec1)
            # plt.show()
            # plt.figure()
            # plt.plot(vec2)
            # plt.show()
            # print([w * LI,w * LI + LW])
        data_f=np.array(data[318:1318])-wavelet
        plt.plot(np.array(data_f))
        # plt.plot(np.array(data[350:1350]),'r')
        # plt.plot(np.array(wavelet), 'g')
        plt.show()

    def plot_psd(self,x):

        plt.psd(x, NFFT=256, Fs=2000, Fc=0, detrend=mlab.detrend_none,
                              window=mlab.window_hanning, noverlap=0)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', default='pufu02.csv',
                        help='file name')
    parser.add_argument('--dwt_wavelet', default='db3',
                        help='dwt_wavelet.')
    parser.add_argument('--cwt_wavelet', default='cgau8',
                        help='cwt_wavelet')
    parser.add_argument('--co_name', default='Noraxon Ultium - EMG3',
                        help='cwt_wavelet')
    parser.add_argument('--fs', default=2000,help='sampling frequency')
    parser.add_argument('--filetree_name', default='20230421XIANGCHUNYA', help='filetree_path')
    parser.add_argument('--filetree_path', default=f'/media/shared/liuruida/database/301/DATA/',
                        help='filetree_path')
    parser.add_argument('--subfiletree_path', default='20230415ABULIKEN',
                        help='subfiletree_path')
    # parser.add_argument('--funcname_arr',default=['aac','rms','mnf','mdf'])
    parser.add_argument('--funcname_arr', default=[ 'rms'])
    # parser.add_argument('--funcname_arr', default=['aac', 'rms', 'skew', 'iav', 'mnf', 'mdf', 'mnp'])
    parser.add_argument('--savefig_path', default='/media/shared/liuruida/database/shoulder_emg/savefig_result/')
    parser.add_argument('--isPatient', default=True)
    parser.add_argument('--task', default='2.PUFU')
    opt = parser.parse_args()
    # print(opt)
    feature_df = pd.DataFrame(columns=np.arange(10))

    for filetree in os.listdir(opt.filetree_path):
        # 读取总文件目录
        # for filename in os.listdir(opt.filetree_path+filetree+'/2.PUFU'):
        #     opt.data_path = opt.filetree_path+filetree+'/2.PUFU'
        #     opt.data_name = filename
        #     # 文件名
        #     result_FIGpath='/media/shared/liuruida/database/shoulder_emg_fig/'+filetree+'/pufu'
        #     # 设置存储位置
        #     if not os.path.exists(result_FIGpath):
        #         # 创建存储文件夹
        #         os.makedirs(result_FIGpath)
        #     opt.savefig_path=result_FIGpath
        #     opt.savefig_path =result_FIGpath
        #     try:
        #         analysis_class = Pufu_analysis(opt)
        #         analysis_class.read_func()
        #         for funcname_arr in opt.funcname_arr:
        #             analysis_class.plot_analysis(funcname_arr)
        #     except Exception as e:
        #         print('error', e)
        #

        for filename in os.listdir(opt.filetree_path+filetree+f'/{opt.task}'):
            if opt.isPatient==True:
                s='H'
            else:
                s='2'

            if filetree[0]==s:
                opt.data_path = opt.filetree_path+filetree+f'/{opt.task}'
                opt.data_name = filename
                opt.subfiletree = filetree
                # result_FIGpath='/media/shared/liuruida/database/shoulder_emg_fig/'+filetree+'/toudan'
                #
                # if not os.path.exists(result_FIGpath):
                #     os.makedirs(result_FIGpath)
                # opt.savefig_path=result_FIGpath
                # opt.savefig_path =result_FIGpath
                # try:
                analysis_class = Pufu_analysis(opt)
                analysis_class.read_func()
                for func_name in opt.funcname_arr:
                    # analysis_class.cwt()
                    feature_series=analysis_class. plot_analysis(func_name)
                    index_name = filetree+'-'+filename
                    feature_df.loc[index_name] = feature_series

    box_arr =[feature_df.iloc[:,i]*1000000 for i in range(10)]

    print(box_arr)
    plt.grid(True)  # 显示网格
    plt.boxplot(box_arr,
                medianprops={'color': 'red', 'linewidth': '1.5'},
                meanline=False,
                showmeans=False,
                meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
                labels=np.arange(1,11))
    plt.ylim((0,500))
    # plt.yticks(np.arange(0.4, 0.81, 0.1))
    # plt.ylabel('Frequency/(Hz)')
    # plt.ylabel('Frequency/(Hz)')
    plt.ylabel('uV')
    plt.xlabel('Channel')
    if opt.isPatient is True:
        title_name='Patient'+'-'+opt.task[2:8]+'-'+opt.funcname_arr[0]
    else:
        title_name = 'Healthy Individuals' + '-'+opt.task[2:8]+'-' + opt.funcname_arr[0]
    plt.title(title_name)
    plt.show()

    print(feature_df)
            # except Exception as e:
            #     print('error', e)

        #
        # for filename in os.listdir(opt.filetree_path+filetree+'/3.TOUDAN/EMG'):
        #     opt.data_path = opt.filetree_path+filetree+'/3.TOUDAN/EMG'
        #     opt.data_name = filename
        #     result_FIGpath='/media/shared/liuruida/database/shoulder_emg_fig/'+filetree+'/toudan'
        #
        #     if not os.path.exists(result_FIGpath):
        #         os.makedirs(result_FIGpath)
        #     opt.savefig_path=result_FIGpath
        #     opt.savefig_path =result_FIGpath
        #     try:
        #         analysis_class = Pufu_analysis(opt)
        #         analysis_class.read_func()
        #         for funcname_arr in opt.funcname_arr:
        #             analysis_class.plot_analysis(funcname_arr)
        #     except Exception as e:
        #         print('error', e)


    print('finished!')
