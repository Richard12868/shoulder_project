# -*- coding: utf-8 -*-
import numpy as np

import matplotlib.pyplot as plt
import sys

print("当前激活的环境:", sys.prefix)
import pywt

# 生成示例信号
t = np.linspace(0, 1, 1000)
signal = np.sin(50 * 2 * np.pi * t) + 0.5 * np.sin(80 * 2 * np.pi * t)

# 进行小波连续变换
waveletname = 'morl'  # 选择小波函数，这里使用Morlet小波
coefficients, frequencies = pywt.cwt(signal, scales=np.arange(1, 31), wavelet=waveletname)

# 绘制结果
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(t, signal)
plt.title('Original Signal')
plt.xlabel('Time')

plt.subplot(122)
plt.imshow(coefficients, extent=[0, 1, 1, 31], cmap='PRGn', aspect='auto')
plt.title('Continuous Wavelet Transform')
plt.xlabel('Time')
plt.ylabel('Scale')
plt.colorbar(label='Magnitude')

plt.tight_layout()
plt.show()



# import pywt
# import numpy as np
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['KaiTi']
#
#
# def cwt(x, fs, totalscal, wavelet='cgau8'):
#     if wavelet not in pywt.wavelist():
#         print('小波函数名错误')
#     else:
#         wfc = pywt.central_frequency(wavelet=wavelet) # 中心频率
#         a = 2 * wfc * totalscal / (np.arange(totalscal, 0, -1)) #尺度因子
#         period = 1.0 / fs
#         [cwtmar, fre] = pywt.cwt(x, a, wavelet, period)
#         amp = abs(cwtmar)
#         return amp, fre


#
#
# def dwt(x, wavelet='db3'):
#     cA, cD = pywt.dwt(x, wavelet, mode='symmetric') #Approximation and detail coefficients.
#     ya = pywt.idwt(cA, None, wavelet, mode='symmetric') # approximated components
#     yd = pywt.idwt(None, cD, wavelet, mode='symmetric') # detail components
#     return ya, yd, cA, cD
#
# if __name__ == '__main__':
#     # 1. 设置数据
#     fs = 1024 #采样率
#     t = np.arange(0, 1.0, 1.0 / fs)
#     f1 = 100
#     f2 = 200
#     f3 = 300
#     f4 = 400
#     data = np.piecewise(t, [t < 1, t < 0.8, t < 0.5, t < 0.3],
#                         [lambda t: 400 * np.sin(2 * np.pi * f4 * t),
#                          lambda t: 300 * np.sin(2 * np.pi * f3 * t),
#                          lambda t: 200 * np.sin(2 * np.pi * f2 * t),
#                          lambda t: 100 * np.sin(2 * np.pi * f1 * t)])
#     plt.figure(1)
#     plt.subplot(2, 2, 1)
#     plt.plot(t,data)
#     plt.ylabel('Amplitude')
#     plt.xlabel('time')
#
#     # 2. 进行连续小波变换
#     amp, fre = cwt(data, fs, 512, 'morl')
#     plt.subplot(2, 2, 2)
#     plt.contourf(t, fre, amp)
#     plt.ylabel('Frequency')
#     plt.xlabel('time')
#
#     # 3. 进行离散小波
#     ya, yd, cA, cD = dwt(data, 'db3')
#     plt.subplot(2, 2, 3)
#     plt.plot(t, ya)
#     plt.xlabel('time')
#     plt.ylabel('近似系数')
#     #plt.show()
#     plt.subplot(2, 2, 4)
#     plt.plot(t, yd)
#     plt.xlabel('time')
#     plt.ylabel('细节系数')
#     plt.show()