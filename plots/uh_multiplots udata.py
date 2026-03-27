import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter

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
import seaborn as sns
from types import SimpleNamespace

import pySPEC as ps
# from pySPEC.time_marching import SWHD_1D, Adjoint_SWHD_1D

from utils import Getfiles

sns.set_style("white")
# palette = sns.color_palette("mako", as_cmap=True)
sns.set_palette(palette='Dark2')

#Nx1 = [1,128]
#multiple_runs=[False,True]
#last_iits = [96,66]
#interps = ['gauss', 'gauss']

Nx1 = [16]
interps = ['gauss']
last_iits = [1]
multiple_runs=[False]

figure_path = './figures'



for i,nx in enumerate(Nx1):
    adjointpath = f'../adjointSW/cases/no_noise'+f'/dx{nx}'

    plt.close('all')
    f1,axs = plt.subplots(nrows = 2, ncols=4, figsize=(20,15)) # originally figsize=(15,10)
    f1.canvas.draw()

    axs1 = axs[0]
    axs2 = axs[-1]

    gf = Getfiles(nx = nx,
                  last_iit= last_iits[i],
                  multiple_runs=multiple_runs[i],
                  normalized_data=True,
                  adjointpath = adjointpath,
                  figure_path=figure_path)
    gf.get_paths()
    gf.get_data()
    gf.get_pinns()
    gf.get_adjoints()

    # Loop over all subplots and plot uu
    for j, ax in enumerate(axs1):
        ax.yaxis.set_major_formatter(FuncFormatter(gf.u_latex_sci_notation))
        ax.set_xlim(gf.domain[0], gf.domain[-1])  # <-- This removes x-axis margin

        # subtract mean of runs
        ax.plot(gf.domain,gf.uu_ref_times[j], label = r'Ground truth', color='black', alpha=0.9)
        ax.plot(gf.domain, gf.uu_adj_times[j] - gf.uu_adj_times[j].mean(), label = r'ASM' , color='green', linestyle='--')
        # subtract mean of runs
        fu = gf.grid.forward(gf.uu_pinns[j])
        fu[0] = 0
        ax.plot(gf.domain, gf.grid.inverse(fu), label = r'PINN', color = 'red')
        ax.set_title(r'$t=$'+f'${gf.tts_tau[j]}$'+ r'$T$', fontsize=24, pad=10)

        # Remove ylabel and y-tick labels from subplots except the first
        if j != 0:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)
        ax.tick_params(labelbottom=False)


    # Match y-axis limits across all subplots
    y_min = min(ax.get_ylim()[0] for ax in axs1)
    y_max = max(ax.get_ylim()[1] for ax in axs1)
    for ax in axs1:
        ax.set_ylim(y_min, y_max)

    axs1[0].legend(fontsize = 22, loc='lower right')
    axs1[0].set_ylabel(r'$u/c$', fontsize = 28)



    # Apply scientific notation formatting
    for i, ax in enumerate(axs2):
        # ax.yaxis.set_major_formatter(FuncFormatter(gf.h_latex_sci_notation))
        ax.set_xlim(gf.domain[0], gf.domain[-1])  # <-- This removes x-axis margin


        ax.plot(gf.domain, gf.hh_ref_times[i], label = r'Ground truth', color='black')
        ax.plot(gf.domain, gf.hh_adj_times[i], label = r'ASM' , color='green',  linestyle = '--')

        h = gf.hh_pinns[i]
        std = gf.hh_stds[i]
        ax.plot(gf.domain, h, label = r'PINN', color = 'red')

    # axs2[0].legend(loc='center right', fontsize = 24)
    axs2[0].set_ylabel(r'$\eta/\eta_0$', fontsize = 28) # no ylabel, only legend

    # Remove ylabel and y-tick labels from subplots except the first
    for i, ax in enumerate(axs2):
        if i != 0:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)

    # Match y-axis limits across all subplots
    y_min = min(ax.get_ylim()[0] for ax in axs2)
    y_max = max(ax.get_ylim()[1] for ax in axs2)
    for ax in axs2:
        ax.set_ylim(y_min, y_max)
    # labels = [r'$(a)$', r'$(b)$']
    # labels = [r'$\text{(a)}$', r'$\text{(b)}$', r'$\text{(c)}$', r'$\text{(d)}$']
    labels1 = [r'$\mathrm{(a)}$', r'$\mathrm{(b)}$', r'$\mathrm{(c)}$', r'$\mathrm{(d)}$']
    labels2 = [r'$\mathrm{(e)}$', r'$\mathrm{(f)}$', r'$\mathrm{(g)}$', r'$\mathrm{(h)}$']

    for i, ax in enumerate(axs1):  # assuming a 1D array of Axes
        ax.text(0.90, 0.95, labels1[i], transform=ax.transAxes,
                ha='center', va='top', fontsize = 28)
        if i==0:
            ax.tick_params(which='both', direction='in', bottom=False, left=True, labelsize= 28)

    for i, ax in enumerate(axs2):  # assuming a 1D array of Axes
        ax.text(0.90, 0.95, labels2[i], transform=ax.transAxes,
                ha='center', va='top', fontsize = 28)
        ax.tick_params(which='both', direction='in', bottom=True, left=False, labelsize= 28)
        ax.set_xlabel('$x/L$', fontsize = 28)
        if i==0:
            ax.tick_params(which='both', direction='in', bottom=True, left=True, labelsize= 28)

    plt.tight_layout()
    if gf.noise:
        plt.savefig(f'{gf.figure_path}/uh_nx{gf.nx}-{gf.std_noise}.pdf')
    else:
        plt.savefig(f'{gf.figure_path}/uh_nx{gf.nx}.pdf')
