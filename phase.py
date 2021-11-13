import numpy as np
import matplotlib.pylab as plt

x = np.linspace(0,2*np.pi, 100)

mol = np.sin(x)**2*(1+np.cos(7.4*x))
iten = np.abs(np.fft.fft(mol))
ax = plt.subplot(111, projection='polar')
ax.plot(x, mol+2)
ax.plot(x, np.ones(mol.shape)*2)
ax.set_rticks([0.5, 1, 1.5, 2]) 
plt.show()

def P_A(x):
    tmp = np.fft.fft(x)
    return np.fft.ifft(tmp/np.abs(tmp)*iten)

def P_B(x):
    tmp = np.real(x)
    tmp[tmp<0] = 0
    return tmp + 0j

x = np.random.rand(100)-0.5 + (np.random.rand(100)-0.5)*0.1j
dxs = []
beta = 1.0
for i in range(5000):
    if (i+1) % 1000 == 0: beta /= 1.02
    pa, pb = P_A(x), P_B(x)
    fa = pa - (pa - x)/beta
    fb = pb + (pb - x)/beta
    dx = beta*(P_A(fb) - P_B(fa))
    dxs.append(np.linalg.norm(np.abs(dx)))
    x += dx

plt.loglog(dxs)
plt.show()

molr = P_B(x)
n_min = 0
val = 1e100
for i in range(len(molr)):
    val_tmp = np.sum((np.roll(molr, i)-mol)**2)
    if val_tmp < val:
        n_min = i
        val = val_tmp
plt.plot(np.roll(molr,n_min))
plt.plot(mol)
plt.show()
