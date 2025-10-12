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

from matplotlib.ticker import FuncFormatter, ScalarFormatter
from types import SimpleNamespace

import pySPEC as ps

from utils import Getfiles

nx = 16
std_noises=[1e-6]
std_names=['1e-6']
std_figure_names=[r'$1\times 10^{-6}$']


interps = ['gauss']
last_iits = [49]

figure_path = './figures'

plt.close('all')


f7,axs7 = plt.subplots(nrows = 1, ncols=4, figsize=(15,5))
axs7=axs7.reshape((1,4))
# Force figure rendering so axis limits are accurate
f7.canvas.draw()


for i, std_noise in enumerate(std_names):
    adjointpath = f'../adjointSW/cases/noise'+f'/dx{nx}_noise{std_names[i]}'


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
    # gf.get_fourier()

    hhms_noise = [gf.hhms_noise[val_step*250] for val_step in gf.tts]

    # Loop over all subplots and plot h
    for j, ax in enumerate(axs7[i]):
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
        ax.plot(gf.domain, gf.hh_ref_times[j], label =  r'$\eta/\eta_0$', color='black', alpha=0.9)
        ax.scatter(gf.domain, hhms_noise[j], label = r'$\eta/\eta_0+\epsilon$', color='blue', alpha=0.2, s=10)
        ax.plot(gf.domain, gf.hh_adj_times[j], label =  r'$\tilde{\eta}/\eta_0$', color='green', linestyle = '--', alpha=1)
        ax.plot(gf.domain, gf.hh_pinns[j], label = r'$\hat{\eta}/\eta_0$', color = 'red', alpha=0.5)

        # Remove ylabel and y-tick labels from subplots except the first
        if j != 0:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)

    # Match y-axis limits across all subplots
    y_min = min(ax.get_ylim()[0] for ax in axs7[0])
    y_max = max(ax.get_ylim()[1] for ax in axs7[0])
    for ax in axs7[i]:
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(gf.domain[0], gf.domain[-1])

for j in range(len(gf.tts)):
    # axs7[0][j].tick_params(labelbottom=False)
    axs7[-1][j].set_xlabel('$x/L$', fontsize = 20)


axs7[0][0].legend(fontsize = 18, loc='center right')
axs7[0][0].set_ylabel(r'$u/c$', fontsize = 20)

# Only show xticks and xlabel for the bottom row of axs

for j in range(len(axs7[0])):
    axs7[0,j].set_title(r'$t=$'+f'${gf.tts_tau[j]}$'+r'$T$', fontsize=20, pad=10)

    labels1 = [r'$\mathrm{(a)}$', r'$\mathrm{(b)}$', r'$\mathrm{(c)}$', r'$\mathrm{(d)}$']
    # labels2 = [r'$\mathrm{(e)}$', r'$\mathrm{(f)}$', r'$\mathrm{(g)}$', r'$\mathrm{(h)}$']

    for i, ax in enumerate(axs7[0]):  # assuming a 1D array of Axes
        ax.text(0.90, 0.95, labels1[i], transform=ax.transAxes,
                ha='center', va='top', fontsize = 20)
        ax.tick_params(which='both', direction='in', bottom=True, left=False, labelsize= 20)
        if i==0:
            ax.tick_params(which='both', direction='in', bottom=True, left=True, labelsize= 20)

f7.tight_layout()

f7.savefig(f'{gf.figure_path}/h_noise-multiplot.pdf')
