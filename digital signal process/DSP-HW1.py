import numpy as np
import matplotlib.pyplot as plt

def filter_FIR(x,f):
	'''
	filter signal {x} with FIR {f}
	'''
	y = np.zeros_like(x)
	for i in range(max(f.keys()),y.shape[0]-1+min(f.keys())):
		for k in f.keys():
			y[i] += f[k] * x[i-k]
	return np.array(y)

def filter_IIR(x,f,g):
	'''
	filter signal {x} with IIR {{f},{g}}
	'''
	y = np.zeros_like(x)
	for i in range(max(f.keys()),y.shape[0]-1):
		for k in f.keys():
			y[i] += f[k] * x[i-k]
		for k in g.keys():
			y[i] -= g[k] * y[i-k]
	return np.array(y)
	
def lowpass_FIR(N_1,N_2,F,delta_t):
	'''
	design lowpass FIR {f}
	N_1 : N_1
	N_2 : N_2
	F : the highest frequence allowed passed
	delta_t : sampling interval
	'''
	f = {}
	f[0] = 2*F*delta_t
	for i in range(-N_1,N_2):
		if i != 0:
			f[i] = np.sin(2*np.pi*F*i*delta_t)/(np.pi*i)
	return f
	
def lowpass_IIR(F,delta_t):
	'''
	design lowpass IIR {f},{g} based on second order Butterworth Function
	F : the highest frequence allowed passed
	delta_t : sampling interval
	'''
	w = np.tan(np.pi*F*delta_t)
	f = {}
	f[0] = f[2] = w**2/(1+2**0.5*w+w**2)
	f[1] = 2*w**2/(1+2**0.5*w+w**2)
	g = {}
	g[1] = -2*(1-w**2)/(1+2**0.5*w+w**2)
	g[2] = (1-2**0.5*w+w**2)/(1+2**0.5*w+w**2)
	return f,g
	
def bandpass_FIR(N_1,N_2,F_2,F_1,delta_t):
	'''
	design bandpass FIR {f}
	N_1 : N_1
	N_2 : N_2
	F_2 : the highest frequence allowed passed
	F_1 : the lowest frequence allowed passed
	delta_t : sampling interval
	'''
	f = {}
	f[0] = 4*((F_2-F_1)/2)*delta_t
	for i in range(-N_1,N_2):
		if i != 0:
			f[i] = 2/np.pi/i*np.sin(2*np.pi*(F_2-F_1)/2*i*delta_t) \
						*np.cos(2*np.pi*(F_2+F_1)/2*i*delta_t)
	return f
	
def bandpass_IIR(F_2,F_1,delta_t):
	'''
	design lowpass IIR {f},{g} based on second order Butterworth Function
	F_2 : the highest frequence allowed passed
	F_1 : the lowest frequence allowed passed
	delta_t : sampling interval
	'''
	w = np.tan(np.pi*(F_2-F_1)*delta_t)
	K = 1+2**0.5*w+w**2
	B = np.cos(np.pi*delta_t*(F_2+F_1))/np.cos(np.pi*delta_t*(F_2-F_1))
	f = {}
	f[0] = f[4] = w**2/K
	f[1] = f[3] = 0
	f[2] = -2*f[0]
	g = {}
	g[1] = -2*B*(2+2**0.5*w)/K
	g[2] = 2*(1+2*B**2-w**2)/K
	g[3] = -2*B*(2-2**0.5*w)/K
	g[4] = (1-2**0.5*w+w**2)/K
	return f,g
	
def plot_fig(t,signal,delta_t,chose,with_noise):
	'''
	plot and save the figure with signal and that after filtering by FIR/IIR
	t : time series
	signal : the raw signal
	delta_t : sampling interval
	chose : 'lowpass' or 'bandpass'
	with_noise : 0->without white noise ; 1->with white noise
	'''
	if with_noise:
		fig_title = 'signal with white noise'
	else:
		fig_title = 'signal without white noise'
	plt.subplot2grid((3,1),(0,0),1,3)
	
	if chose == 'lowpass':
		FIR_f = lowpass_FIR(10,10,10,delta_t)
		IIR_f,IIR_g = lowpass_IIR(10,delta_t)
	elif chose == 'bandpass':
		FIR_f = bandpass_FIR(10,10,60,40,delta_t)
		IIR_f,IIR_g = bandpass_IIR(60,40,delta_t)
		
	for i in range(1,4):
		plt.subplot(3,1,i)
		if i == 1:
			y = filter_FIR(signal,FIR_f)
			plt.text(1.01,2.2,'FIR')
			plt.title(fig_title)
		elif i == 2:
			y = filter_IIR(signal,IIR_f,IIR_g)
			plt.text(1.01,2.2,'IIR_1')
		elif i == 3:
			y = filter_IIR(signal,IIR_f,IIR_g)
			y = filter_IIR(y,IIR_f,IIR_g)
			plt.text(1.01,2.2,'IIR_2')
			
		plt.plot(t,signal,color='black',linestyle=':',label='raw signal')
		plt.plot(t,y,color='red',label='signal after filter')
		plt.xlim(0,1)
		plt.ylim(-3,3)

	plt.legend(loc='center',bbox_to_anchor=(0.5, -0.3),ncol=2)
	plt.savefig(fig_title+' under '+chose,dpi=300)
	plt.show()

def plot_raw_signal(t,signal):
	'''
	plot and save the figure of given signals
	t : time series
	signal : the given signal
		signal[0] : the given signal without white noise
		signal[1] : the given signal with white noise
	'''
	plt.subplot(3,1,1)
	plt.plot(t,signal[0],color='black')
	plt.title('signal without white noise')
	plt.xlim(0,1)
	plt.ylim(-3,3)
	plt.subplot(3,1,3)
	plt.plot(t,signal[1],color='black')
	plt.title('signal with white noise')
	plt.xlim(0,1)
	plt.ylim(-3,3)
	plt.savefig('raw_signal',dpi=300)
	plt.show()

	
if __name__ == '__main__':
	'''
	signal : the given signal
	t : time series, from 0s to 1s with sampling interval delta_t
	delta_t : sampling interval, unit:s

	when plot the given signal
		use code : plot_raw_signal(t,signal)
	when plot the signal after filter
		use code : plot_fig(t,signal[0],delta_t,'lowpass',0)
			which signal[0] means input the given signal without white noise
				  'lowpass' means using lowpass filter FIR/IIR
	'''
	delta_t = 0.001
	t = np.arange(0,1,delta_t)
	signal = []
	signal.append(np.sin(2*np.pi*10*t)
				  +np.sin(2*np.pi*50*t)
				  +np.sin(2*np.pi*100*t))
	signal.append(signal[0]+0.6*np.random.randn(t.shape[0]))
	#plot_raw_signal(t,signal)
	#plot_fig(t,signal[1],delta_t,'lowpass',1)
	#plot_fig(t,signal[1],delta_t,'bandpass',1)