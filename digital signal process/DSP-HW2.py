import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def show_problem():
    '''
        show the curve of raw signal (displacement and velocity)
    '''
    data = np.loadtxt('data.txt')
    time_series = np.arange(0,data.shape[0]*0.0005,0.0005)
    velocity = (data[1:,]-data[:-1,])/0.0005
    plt.figure(figsize=(8, 6))
    plt.subplot(2,1,1)
    plt.plot(time_series,data,color='black')
    plt.ylim([120,220])
    plt.xlim([0,60])
    plt.ylabel('position(mm)')
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(time_series[1:],velocity,color='black')
    plt.xlim([0,60])
    plt.ylabel('velocity(mm/s)')
    plt.xlabel('time(s)')
    plt.grid(True)
    plt.savefig('fig_1',dpi=300)
    plt.show()

def FFT_analysis():
    '''
        show the frequency of displacement
    '''
    data = np.loadtxt('data.txt')
    fft_data = np.fft.fft(data)
    abs_fft_data = abs(fft_data)/(len(data)/2)
    frequency = np.linspace(0,1000,len(data)/2)
    plt.figure(figsize=(8, 6))
    plt.plot(frequency,abs_fft_data[0:int(len(data)/2)],color='black')
    plt.ylabel('frequency(Hz)')
    plt.xlabel('amplitude')
    plt.grid(True)
    plt.savefig('fig_2',dpi=300)
    plt.show()

def lowpass():
    '''
        make a lowpass filter (IIR)
        and filtered the displacement
        then get the velocity
    '''
    data = np.loadtxt('data.txt')
    b,a = signal.butter(4,0.0005,'low')
    filtered_data = signal.filtfilt(b,a,data)  
    velocity = (filtered_data[1:,]-filtered_data[:-1,])/0.0005
    time_series = np.arange(0,filtered_data.shape[0]*0.0005,0.0005)
    plt.figure(figsize=(8, 6))
    plt.subplot(2,1,1)
    plt.plot(time_series,filtered_data,color='black')
    plt.ylim([120,220])
    plt.xlim([0,55])
    plt.ylabel('position(mm)')
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(time_series[1:],velocity,color='black')
    plt.xlim([0,55])
    plt.ylabel('velocity(mm/s)')
    plt.xlabel('time(s)')
    plt.grid(True)
    plt.savefig('fig_3',dpi=300)
    plt.show()

if __name__ == '__main__':
    show_problem()
    FFT_analysis()
    lowpass()