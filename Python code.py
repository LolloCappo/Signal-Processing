import numpy as np

## FFT
data = #data to analyze
dt = #time step
n = data.shape[1] 
sampling_frequency = 1/dt

FFT = np.fft.rfft(data, n=n, axis=1) * 2/n
freq = np.fft.rfftfreq(n, dt)

plt.plot(freq,np.abs(FFT))

##STFT
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
mod = 500*np.cos(2*np.pi*0.25*time)
carrier = amp * np.sin(2*np.pi*3e3*time + mod)
noise = np.random.normal(scale=np.sqrt(noise_power),
                         size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noise

f, t, Zxx = signal.stft(x, fs, nperseg=1000)
plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()



## FRF
import pyFRF

FRF = pyFRF.FRF(sampling_freq=sampling_frequency, 
                n_averages=force.shape[0], 
                fft_len=force.shape[1], 
                exc_type='f', 
                resp_type='a')
                
                
for i in range(force.shape[0]):
    FRF.add_data(force[i], acceleration[i])
    
receptance = FRF.get_FRF(form='receptance')
frequency = FRF.get_f_axis()

fig, ax = plt.subplots(2, figsize=(20, 8))

ax[0].semilogy(frequency, np.abs(receptance))
ax[1].plot(frequency, np.angle(receptance))

ax[0].set_xlim(0, 2000), ax[1].set_xlim(0, 2000)
ax[1].set_xlabel('Frequency [Hz]')
ax[0].set_ylabel(r'|Receptance [m/N]|')
ax[1].set_ylabel('Phase [$^\circ$]');
