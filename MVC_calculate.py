import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert
import numpy as np
import pandas as pd
import scipy.io as scio
from scipy import signal
from MVC_func import *

import pickle
import warnings
warnings.filterwarnings("ignore")

class MVCvalue:
    co = np.arange(1, 15)
    # 对应的文件
    muscles = ['gongertouji', 'gongsantouji', 'xiongdaji', 'xiefangji', 'sanjiaoji', 'gangxiadayuan', 'beikuoqianju']

    # 对应的列
    mudic = {'gongertouji':[3], 'gongsantouji':[4], 'xiongdaji':[5], 'xiefangji':[6], 'sanjiaoji':[7,8,9], 'gangxiadayuan':[10], 'beikuoqianju':[11,12]}


    def __init__(self,  path):

        # 读取的文件目录
        self.path = path


    def reframe(self):
        reFrame = pd.DataFrame(data=np.ones((6, 10)), columns=['gongertouji', 'gongsantouji', 'xiongdaji', 'xiefangji',
                                                               'sanjiaojiqian', 'sanjiaojizhong', 'sanjiaojihou', 'gangxiaji', 'beikuoji', 'qianjuji'])
        for s in MVCvalue.muscles:
            # 对应文件
            # mu_name = self.name+'-' + s + '-MVC.xlsx'

            # 文件对应的列
            arr = MVCvalue.mudic[s]


            for j in arr:
                # 文件对应的列
                mu_name = 'mvc-' + s + '-01' + '.csv'
                try:

                    datalist=mvc_dataprocess(self.path+mu_name,j)
                    t = np.arange(len(datalist))
                    # plt.figure()
                    # plt.plot(t, datalist)
                    # plt.title(s+str(j))
                    # plt.show()

                    reFrame.iloc[0,j-3]=max(datalist)
                except Exception as e:
                    print(e)
                    print('error',mu_name)
                mu_name = 'mvc-' + s + '-02' + '.csv'
                try:

                    datalist = mvc_dataprocess(self.path + mu_name,j)
                    t = np.arange(len(datalist))
                    # plt.figure()
                    # plt.plot(t, datalist)
                    # plt.title(s + str(j))
                    # plt.show()

                    reFrame.iloc[1, j - 3] = max(datalist)
                except Exception as e:
                    print(e)
                    print('error', mu_name)
                mu_name = 'mvc-' + s + '-03' + '.csv'
                try:

                    datalist = mvc_dataprocess(self.path + mu_name,j)
                    t = np.arange(len(datalist))
                    # plt.figure()
                    # plt.plot(t, datalist)
                    # plt.title(s + str(j))
                    # plt.show()

                    reFrame.iloc[2, j - 3] = max(datalist)
                except Exception as e:
                    print(e)
                    print('error', mu_name)



        print(reFrame)

        # reFrame.to_excel(pathFrame)

if __name__ == '__main__':
    # 求MVC列表
    MVCdirpath= '/media/shared/liuruida/database/shoulder_emg/data/shiji/mvc/'
    chaiMVCvalues=MVCvalue(MVCdirpath)
    # 保存目录
    # MVCsavepath = '/media/shared/liuruida/database/shoulder_emg/data/shiji/shiji-sdMVC.xls'
    chaiMVCvalues.reframe()