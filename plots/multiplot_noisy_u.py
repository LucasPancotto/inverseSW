import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from types import SimpleNamespace

import pySPEC as ps
# from pySPEC.time_marching import SWHD_1D, Adjoint_SWHD_1D

from utils import Getfiles

nx = 16
std_noises=[1e-7,5e-7,1e-6,5e-6] #,1e-5]
std_names=['1e-7','5e-7','1e-6','5e-6'] #,'1e-5']
std_figure_names=[r'$1\times 10^{-7}$',r'$5\times 10^{-7}$',r'$1\times 10^{-6}$',r'$5\times 10^{-6}$',r'$1\times 10^{-5}$']

interps = ['gauss', 'gauss','gauss','gauss', 'gauss']
last_iits = [62,50,49,52,37]
figure_path = './figures'

plt.close('all')
f,axs = plt.subplots(nrows = len(std_noises), ncols=4, figsize=(15,15))

# Force figure rendering so axis limits are accurate
f.canvas.draw()


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
    gf.get_fourier()

    if gf.multiple_runs:
        mean_pinns = np.array(gf.pinn_times).mean(0)
        mean_pinn = np.array(gf.pinn).mean(0)
    else:
        mean_pinns = np.array(gf.pinn_times)
        mean_pinn = np.array(gf.pinn)

    hhms_noise = gf.hhms_noise

    plt.close('all')

    # Loop over all subplots and plot uu
    for j, ax in enumerate(axs[i]):
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
        ax.plot(gf.domain, gf.uu_ref_times[j], label = '$u$', color='black', alpha=0.9)
        ax.plot(gf.domain, gf.uu_adj_times[j], label = 'Adjoint $\hat{u},\; \epsilon~$'+f'{std_figure_names[i]}' , color='green', linestyle = '--', alpha=0.5)
        # plot mean of runs
        fu = gf.grid.forward(mean_pinns[j][0])
        fu[0] = 0
        ax.plot(gf.domain, gf.grid.inverse(fu), label = 'PINN $\hat{u},\; \epsilon~$'+f'{std_figure_names[i]}', color = 'red', linestyle = '--', alpha=0.5)
        ax.set_title(f'$t=${gf.tts[j]}')

        # Remove ylabel and y-tick labels from subplots except the first
        if j != 0:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)

    # Match y-axis limits across all subplots
    y_min = min(ax.get_ylim()[0] for ax in axs[i])
    y_max = max(ax.get_ylim()[1] for ax in axs[i])
    for ax in axs[i]:
        ax.set_ylim(y_min, y_max)

    axs[i][0].legend(fontsize = 20)
    axs[i][0].set_ylabel('$u$', fontsize=20)


# Only show xticks and xlabel for the bottom row of axs
for i in range(len(std_noises)):
    for j in range(4):
        if i != len(std_noises) - 1:
            axs[i][j].tick_params(labelbottom=False)
            axs[i][j].set_xlabel('')
        else:
            axs[i][j].set_xlabel('$x$', fontsize=20)

f.tight_layout()

f.savefig(f'{gf.figure_path}/u_noise-multiplot.pdf')
