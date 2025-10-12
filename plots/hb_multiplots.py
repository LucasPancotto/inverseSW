import os
import json
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
 "text.latex.preamble": r"""
        \usepackage{amsmath}
        \usepackage{mathptmx}
        \usepackage{bm}
    """
})


from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from types import SimpleNamespace

import pySPEC as ps
# from pySPEC.solvers import SWHD_1D, Adjoint_SWHD_1D

from utils import Getfiles

# Nx1 = [1,128]
Nx1 = [1,128]
last_iits = [96,66]
interps = ['gauss', 'gauss']
multiple_runs=[False,True]
plot_measurements = [False,True]

# figure_path = 'no_noise_figures'
figure_path = '/home/lpancotto/code/tesis/adjoint/results/inverse_hb_h0_u0/paper_figures'

plt.close('all')
f1,axs1 = plt.subplots(nrows = len(Nx1), figsize=(15,10))

# Force figure rendering so axis limits are accurate
f1.canvas.draw()

for i,nx in enumerate(Nx1):
    adjointpath = f'/home/lpancotto/code/tesis/adjoint/results/inverse_hb_h0_u0/dx{nx}'+f'/dx{nx}_dt1_inv_hb_h0_u0-nf_{interps[i]}'
    gf = Getfiles(nx = nx,
                  last_iit= last_iits[i],
                  multiple_runs=multiple_runs[i],
                  plot_measurements=plot_measurements[i],
                  normalized_data=True,
                  adjointpath = adjointpath,
                  figure_path=figure_path)
    gf.get_paths()
    gf.get_data()
    gf.get_pinns()
    gf.get_adjoints()

    axs1[i].plot(gf.domain,gf.ref[-1] , color = 'black' , label = r'$h_b/h_0$', alpha = 1)
    axs1[i].plot(gf.domain, gf.adj[-1] , label = r'$\tilde{h}_b/h_0$'  , linestyle='--', color = 'green')
    axs1[i].plot(gf.domain, gf.hb_pinn , label = r'$\hat{h}_b/h_0$', alpha = 0.7  , color = 'red')
    if gf.multiple_runs:
        plt.fill_between(gf.domain, gf.hb_pinn - gf.hb_std, gf.hb_pinn + gf.hb_std, alpha=0.3) # , label= '$\hat{h_b}\pm$ $\sigma(\hat{h_b})$')
    axs1[i].set_xlim(gf.domain[0], gf.domain[-1])  # <-- This removes x-axis margin

    axs1[i].set_ylabel(r'$h_b/h_0$', fontsize=20)

    if gf.plot_measurements:
        if gf.normalized_data:
            N = 1024
            x_min, x_max = 0, 2 * np.pi/gf.L
            x = np.linspace(x_min, x_max, N, endpoint=False)
            stride = gf.nx
            x_markers = x[::stride]
            y_loc = -0.01
            y_markers = np.zeros_like(x_markers)+y_loc
        else:
            N = 1024
            x_min, x_max = 0, 2 * np.pi
            x = np.linspace(x_min, x_max, N, endpoint=False)
            stride = gf.nx
            x_markers = x[::stride]
            y_loc = -0.01
            y_markers = np.zeros_like(x_markers)+y_loc

        axs1[i].plot(x_markers, y_markers, 'x', color='orange' )# , label='data point locations')

axs1[0].legend(fontsize=20, loc='upper left')

# Only show xticks and xlabel for the bottom row of axs
for ax in axs1[:-1]:
    ax.tick_params(labelbottom=False)
axs1[-1].set_xlabel(r'$x/L$', fontsize=20)


axs1[0].tick_params(which='both', direction='in', bottom=False, left=True, labelsize = 20)
axs1[-1].tick_params(which='both', direction='in', bottom=True, left=True, labelsize = 20)
# labels = [r'$(a)$', r'$(b)$']
# labels = [r'$\text{(a)}$', r'$\text{(b)}$', r'$\text{(c)}$', r'$\text{(d)}$']
labels = [r'$\mathrm{(a)}$', r'$\mathrm{(b)}$']

for i, ax in enumerate(axs1):  # assuming axs3 is a 1D array of Axes
    ax.text(0.95, 0.9, labels[i], transform=ax.transAxes,
            ha='center', va='top', fontsize=18)

f1.tight_layout()

f1.savefig(f'{gf.figure_path}/hb_nx-1_128-multiplot.pdf')
