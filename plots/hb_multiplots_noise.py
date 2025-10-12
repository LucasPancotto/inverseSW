import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from types import SimpleNamespace

import pySPEC as ps
# from pySPEC.time_marching import SWHD_1D, Adjoint_SWHD_1D

from utils import Getfiles

plt.rcParams.update({"text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
 "text.latex.preamble": r"""
        \usepackage{amsmath}
        \usepackage{mathptmx}
        \usepackage{bm}
    """
})

nx = 16
std_noises=[1e-7, 5e-6]
std_names=['1e-7', '5e-6']
std_figure_names=[r'$1\times 10^{-7}$',r'$5\times 10^{-6}$']


interps = ['gauss',  'gauss']
last_iits = [64,52]
# multiple_runs=[False,False,False,True]

figure_path = '/home/lpancotto/code/tesis/adjoint/results/inverse_hb_h0_u0/noise_cases/paper_figures'

plt.close('all')

f2,axs2 = plt.subplots(nrows = len(std_noises), figsize=(15,10))


# Force figure rendering so axis limits are accurate
f2.canvas.draw()

for i, std_noise in enumerate(std_names):
    adjointpath = f'/home/lpancotto/code/tesis/adjoint/results/inverse_hb_h0_u0/noise_cases/gauss_interpolation'+f'/dx{nx}_dt1_inv_hb_h0_u0-nf_{interps[i]}-noise{std_names[i]}'
    gf = Getfiles(nx = nx,
                  last_iit= last_iits[i],
                  adjointpath = adjointpath,
                  noise=True,
                  std_noise=std_noise,
                  multiple_runs=True,
                    normalized_data=True,
                  figure_path=figure_path)
    gf.get_paths()
    gf.get_data()
    gf.get_pinns()
    gf.get_adjoints()

    hhms_noise = gf.hhms_noise
    if gf.multiple_runs:
        plt.fill_between(gf.domain, gf.hb_pinn -  gf.hb_std, gf.hb_pinn + gf.hb_std,color='blue', alpha=0.2) # , label= '$\hat{h_b}\pm$ $\sigma(\hat{h_b})$')
    axs2[i].plot(gf.domain,gf.ref[-1] , color = 'black' , label = r'$h_b/h_0$', alpha = 1)
    axs2[i].plot(gf.domain, gf.adj[-1] , label = r'$\tilde{h}_b/h_0$'   , linestyle='--', color = 'green')
    axs2[i].plot(gf.domain, gf.hb_pinn , label =r'$\hat{h}_b/h_0$', alpha = 0.7  , color = 'red')
    axs2[i].set_ylabel(r'$h_b/h_0$', fontsize=20)
    axs2[0].legend(loc = 'upper left', fontsize=20)

    # Match y-axis limits across all subplots
    y_min = min(ax.get_ylim()[0] for ax in axs2)
    y_max = max(ax.get_ylim()[1] for ax in axs2)
    for ax in axs2:
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(gf.domain[0], gf.domain[-1])

for ax in axs2[:-1]:
    ax.tick_params(labelbottom=False)
axs2[-1].set_xlabel(r'$x/L$', fontsize=20)

axs2[0].tick_params(which='both', direction='in', bottom=False, left=True, labelsize = 20)
axs2[-1].tick_params(which='both', direction='in', bottom=True, left=True, labelsize = 20)
labels = [r'$\mathrm{(a)}$', r'$\mathrm{(b)}$', r'$\mathrm{(c)}$', r'$\mathrm{(d)}$']

for i, ax in enumerate(axs2):  # assuming axs3 is a 1D array of Axes
    ax.text(0.95, 0.9, labels[i], transform=ax.transAxes,
            ha='center', va='top', fontsize=18)


f2.tight_layout()

f2.savefig(f'{gf.figure_path}/hb_noise-1e-7_5e-6-multiplot.pdf')
