import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from types import SimpleNamespace
import os

import pySPEC as ps
from pySPEC.solvers.swhd_1d import SWHD_1D
from pySPEC.solvers.adjoint_swhd_1d import Adjoint_SWHD_1D

class Generator():
    """
    Generates initial conditions for pseudospectral integration.
    """

    def __init__(self,
                current_dir,
                hb_noise=True,
                uh_noise=False):

        self.current_dir = current_dir
        pm = json.load(open(f'{self.current_dir}/params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
        pm.Lx = 2*np.pi*pm.Lx
        self.grid = ps.Grid1D(pm)
        self.xx = self.grid.xx
        self.Nx = 1024
        self.h0 = pm.h0

        self.out_path = pm.out_path

        self.hb_noise = hb_noise
        self.uh_noise = uh_noise
        self.hb = None
        self.uu = None
        self.hh = None

    def noise(self, kmin , kmax, Amin=False, Amax=False):
        ks = np.random.uniform(kmin,kmax, kmax-kmin)
        if ((Amin == False) or (Amax==False)):
            ls = (1024/ks)*40*5e-5 # (1024/k)*40m*5e-5amps/m : conversion de ks a sus amplitudes correspondientes
            As = ls*np.random.uniform(1/4, 1/2, len(ls))
        else:
            As = np.random.uniform(Amin, Amax, len(ks))
        phis = np.random.uniform(0,np.pi, len(ks))
        noise = 0
        for k,A,phi in zip(ks,As,phis):
            mode = A*np.sin(self.grid.xx*k + phi)
            noise = noise + mode
        noise[0] = 0
        noise = noise/len(ks)
        return noise

    def get_hb(self):
        Nx =  self.Nx
        dx = self.grid.dx

        # structures of l ~ 400m to 1000m, scaled: 0.02 to 0.05  --> k ~ 100 to 40
        if self.hb_noise:
            noisek100 = self.noise(kmin = 40 , kmax = 100)
        else:
            noisek100 = 0
        # structures of l ~ 80m to 100m, scaled: 0.004 to 0.005  --> k ~ 500 to 400
        # noisek500 = self.noise(grid, kmin = 400 , kmax = 500, Amin = 0.001, Amax= 0.005  ) # not seen by pseudospectral

        s0 =  0.1
        s1 =  0.3
        s2 = 1.4
        s3 = 0.05
        s4 = 0.2
        s5 = 0.8
        hb = s0*np.exp(-(self.grid.xx-np.pi/s2)**2/s1**2) + s3*np.exp(-(self.grid.xx-np.pi/s5)**2/s4**2) + noisek100
        self.hb = hb
        return hb

    def get_uh(self):
        Nx =  self.Nx
        dx = self.grid.dx

        # structures of l ~ 400m to 1000m, scaled: 0.02 to 0.05  --> k ~ 100 to 40
        if self.uh_noise:
            noisek100uu = self.noise(kmin = 40 , kmax = 100, Amin = 0.00001, Amax= 0.000025  )
        else:
            noisek100uu = 0

        # Initial conditions
        v1 = 0.00025
        v2 =  0.5
        v3 = 2
        uu = v1 * np.exp(-((self.grid.xx - np.pi/v3) ** 2) / v2 ** 2) + noisek100uu

        c1 = 5e-5
        c2 = 0.5
        c3 = 2
        hh = self.h0 + c1 * np.exp(-((self.grid.xx - np.pi/c3) ** 2) / c2 ** 2)

        self.uu = uu
        self.hh = hh

        return uu,hh

    def plot_fields(self):
        plt.figure( figsize = (15,5))
        plt.plot(self.grid.xx , self.hb , label = "$h_b(x)")
        plt.legend()
        plt.savefig(f'{self.out_path}/hb.png')
        plt.figure( figsize = (15,5))
        plt.plot(self.grid.xx , self.hh , label = "$h_0(x)")
        plt.legend()
        plt.savefig(f'{self.out_path}/hh0.png')
        plt.figure( figsize = (15,5))
        plt.plot(self.grid.xx , self.uu , label = "$u_0(x)")
        plt.legend()
        plt.savefig(f'{self.out_path}/uu0.png')

        # Compute Fourier Transform of hb
        hb_fft = np.fft.fft(self.hb)
        hb_freq = np.fft.fftfreq(self.Nx, d=self.grid.dx)

        # Take only the positive frequencies for plotting
        hb_amplitude = np.abs(hb_fft[:self.Nx // 2])
        hb_freq_positive = hb_freq[:self.Nx // 2]

        # Plot the Fourier Amplitude Spectrum
        plt.figure(figsize=(10, 6))
        plt.loglog(hb_freq_positive, hb_amplitude, color='blue', label = '$fft(h_b)$')
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.title('Fourier Amplitude Spectrum of hb')
        plt.legend()
        plt.grid()
        plt.savefig(f'{self.out_path}/fft_hb.png')
