import numpy as np
import matplotlib.pyplot as plt

ref = np.load('ref.npy')
pinn = np.load('predicted.npy')

loss = np.loadtxt('output.dat', unpack=True)
plt.figure(0)
plt.semilogy(loss[0], loss[1])
plt.semilogy(loss[0], loss[2])

plt.figure(1)
plt.plot(ref[0])
plt.plot(pinn[0])

plt.figure(2)
plt.plot(ref[3])
plt.plot(pinn[3])

plt.show()
