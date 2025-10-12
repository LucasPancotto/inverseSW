import os
import json
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,
    "font.family": "serif",
    # "font.serif": ["Times"],
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
max_eta = 5e-5
std_noises=[1e-7, 5e-7, 1e-6, 5e-6]
std_names=['1e-7', '5e-7', '1e-6','5e-6']
normalized_std_noises = np.array(std_noises)/max_eta

std_figure_names=[r'$1\times 10^{-7}$',r'$5\times 10^{-7}$',r'$1\times 10^{-6}$',r'$5\times 10^{-6}$']

interps = ['gauss', 'gauss','gauss','gauss']
last_iits = [62,50,49,52]
figure_path = './figures'

# sweep error in different scales for different noise amplitudes
sweep_pinn = []
sweep_adj = []
sweep_pinn_0 = []
sweep_adj_0 = []
sweep_pinn_mid = []
sweep_adj_mid = []

Usweep_pinn = []
Usweep_adj = []
Usweep_pinn_0 = []
Usweep_adj_0 = []
Usweep_pinn_mid = []
Usweep_adj_mid = []



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

    # gf.plot_fields()
    # gf.plot_losses()
    gf.get_fourier()
    # gf.plot_fourier()
    # gf.plot_fourier_error()
    d = gf.get_integral_error()
    Epinn, Eadj, Epinn_0, Eadj_0, Epinn_mid, Eadj_mid = d['hb_errors']
    UEpinn, UEadj, UEpinn_0, UEadj_0, UEpinn_mid, UEadj_mid = d['u_errors']
    sweep_pinn.append(Epinn)
    sweep_adj.append(Eadj)

    sweep_pinn_0.append(Epinn_0)
    sweep_adj_0.append(Eadj_0)

    sweep_pinn_mid.append(Epinn_mid)
    sweep_adj_mid.append(Eadj_mid)

    Usweep_pinn.append(UEpinn)
    Usweep_adj.append(UEadj)

    Usweep_pinn_0.append(UEpinn_0)
    Usweep_adj_0.append(UEadj_0)

    Usweep_pinn_mid.append(UEpinn_mid)
    Usweep_adj_mid.append(UEadj_mid)

print('normalized_std_noises, ', r'$\epsilon/\eta_0$', normalized_std_noises)

plt.close('all')
f1,axs1 = plt.subplots(nrows = 2, figsize=(15,10))
# Force figure rendering so axis limits are accurate
f1.canvas.draw()


axs1[0].semilogx(normalized_std_noises, sweep_pinn, linestyle='-', marker='o', label=r'$\mathcal{E}_{\hat{h}_b}$', color = 'r')
axs1[0].semilogx(normalized_std_noises, sweep_adj, linestyle='--', marker='s', label=r'$\mathcal{E}_{\tilde{h}_b}$', color = 'g')

axs1[0].set_xlabel(r'$\epsilon$', fontsize=22)
axs1[0].set_ylabel(r'$\mathcal{E}_{h_b}$', fontsize=25)

axs1[0].xaxis.set_major_formatter(FuncFormatter(gf.u_latex_sci_notation))

axs1[0].legend( fontsize=22)
axs1[0].tick_params(labelbottom=False)
axs1[0].set_xlabel('')


axs1[1].semilogx(normalized_std_noises, Usweep_pinn, linestyle='-', marker='o', label=r'$\mathcal{E}_{\hat{u}}$', color = 'r')
axs1[1].semilogx(normalized_std_noises, Usweep_adj, linestyle='--', marker='s', label=r'$\mathcal{E}_{\tilde{u}}$', color = 'g')

axs1[1].set_xlabel(r'$\epsilon$', fontsize=22)
axs1[1].set_ylabel(r'$\mathcal{E}_{u}$', fontsize=25)
axs1[1].xaxis.set_major_formatter(FuncFormatter(gf.u_latex_sci_notation))
# axs1[1].legend( fontsize=22)

axs1[1].set_xticks(normalized_std_noises)

axs1[0].tick_params(which='both', direction='in', bottom=False, left=True, labelsize = 2)
axs1[-1].tick_params(which='both', direction='in', bottom=True, left=True, labelsize = 20)
axs1[-1].tick_params(axis='x' , which='both', direction='in', bottom=True, left=False , labelsize = 20)

labels = [r'$\mathrm{(a)}$', r'$\mathrm{(b)}$']

for i, ax in enumerate(axs1):  # assuming axs3 is a 1D array of Axes
    ax.text(0.98, 0.95, labels[i], transform=ax.transAxes,
            ha='center', va='top', fontsize=18)

plt.tight_layout()
plt.savefig(f'{gf.figure_path}/normalized_total_error-hb_u-noise.pdf')
