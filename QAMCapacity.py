import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import *
try:
    import cPickle as pickle
except ImportError:
    import pickle


N = 16 # degree of approximation
M = 16 # M-QAM
x, w = hermgauss(N)
const = np.array([a + 1j * b for a in np.linspace(-3, 3, 4) for b in np.linspace(3, -3, 4)]) / np.sqrt(10) # Normalized constellation fixed for 16-QAM


def dB2gain(dB):
    return 10 ** (dB / 10.0)


def gain2dB(gain):
    return 10*np.log10(gain)


def qam_cap(SNR):
    # SNR in gain
    # N = 16  # degree of approximation
    # M = 16  # M-QAM
    # x, w = hermgauss(N)
    # const = np.array([a + 1j * b for a in np.linspace(-3, 3, 4) for b in
    #                   np.linspace(3, -3, 4)]) / np.sqrt(
    #     10)  # Normalized constellation fixed for 16-QAM
    cap = np.empty(M)
    blah = np.empty([N, N])
    for x1 in range(M):
        for m1 in range(N):
            for m2 in range(N):
                what = 0
                for x2 in range(M):
                    what += np.exp(-np.abs(np.sqrt(SNR) * (const[x1] - const[x2]) + x[m1] + (1j * x[m2])) ** 2 + (x[m1] ** 2) + (x[m2] ** 2))
                blah[m1, m2] = w[m1] * w[m2] * np.log2(what) / np.pi
        cap[x1] = np.sum(blah)
    totcap = np.log2(M) - sum(cap) / M
    return totcap


t1 = np.linspace(0, 1e-5, 5000)
t2 = np.linspace(-50, 30, 5000)
t2 = [dB2gain(_) for _ in t2]
t = np.append(t1, t2)
y = [qam_cap(i) for i in t]
# plt.plot(t,y,'ro')
# plt.show()
f_linear = interp1d(t, y)

with open('interpolator.pkl', 'wb') as f:
    pickle.dump(f_linear, f)
with open('interpolator.pkl', 'rb') as f:
    f_loaded = pickle.load(f)

print(f_loaded)
