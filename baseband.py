from scipy import signal
import numpy as np
import random
import matplotlib.pyplot as plt

bits=25
rand_bit_string = np.random.randint(2, size=bits)
tmin = 0
tmax = 1
samples = 500
freq = 50
samples_per_bit = int(samples/bits)
t = np.linspace(tmin, tmax, samples, endpoint=False)
bin_signal = rand_bit_string.repeat(samples_per_bit)
ask = np.sin(2*np.pi*freq*t*bin_signal)
psk = np.sin((2*np.pi*freq*t+np.pi*bin_signal))
plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(t, bin_signal)
plt.subplot(3, 1, 2)
plt.plot(t, ask)
plt.subplot(3, 1, 3)
plt.plot(t, psk)

spec = (np.absolute(np.fft.fft(bin_signal)) ** 2) / samples_per_bit
plt.figure(2)
plt.plot(spec)
plt.Axes.set_xlim(left=-1, right=1)

plt.show()
