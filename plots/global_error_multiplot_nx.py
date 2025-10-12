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

# sweep error in different scales for different nx
sweep_pinn = []
sweep_adj = []
sweep_pinn_0 = []
sweep_adj_0 = []
sweep_pinn_mid = []
sweep_adj_mid = []
sweep_pinn_max = []
sweep_adj_max = []

Usweep_pinn = []
Usweep_adj = []
Usweep_pinn_0 = []
Usweep_adj_0 = []
Usweep_pinn_mid = []
Usweep_adj_mid = []

Nx1 = [1,16,64]
interps = ['gauss', 'gauss','gauss']
last_iits = [96,81,56]
multiple_runs=[False,False,False]

figure_path = './figures'

for i,nx in enumerate(Nx1):
    adjointpath = f'../adjointSW/cases/no_noise'+f'/dx{nx}'

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
    gf.get_fourier()


    # gf.plot_fields()
    # gf.plot_losses()
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



# pinns with multiple runs
# Nx2 = [128,256]
Nx2 = [128]
interps = ['gauss']
last_iits = [66]
multiple_runs=[True]


for i,nx in enumerate(Nx2):
    adjointpath = f'/home/lpancotto/code/tesis/adjoint/results/inverse_hb_h0_u0/dx{nx}'+f'/dx{nx}_dt1_inv_hb_h0_u0-nf_{interps[i]}'

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
    gf.get_fourier()

    # gf.plot_fields()
    # gf.plot_losses()
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

Nx=Nx1+Nx2
dx = np.array(Nx)/gf.Nl
dx_labels = [r'$\frac{3}{1024}$', r'$\frac{3}{64}$', r'$\frac{3}{16}$', r'$\frac{3}{8}$']
print('Nl = ' , gf.Nl, '\n')
print('$\delta_x = $ ', dx, '\n')
print('$\delta_x labels = $ ', dx_labels, '\n')


figure_path = figure_path
gf = Getfiles(nx = 1,figure_path=figure_path, multiple_runs=False) # just for figure path

###########


plt.close('all')
f1,axs1 = plt.subplots(nrows = 2, figsize=(15,10))

# Force figure rendering so axis limits are accurate
f1.canvas.draw()


axs1[0].plot(dx, sweep_pinn, linestyle='-', marker='o', label=r'$\mathcal{E}_{\hat{h}_b}$', color = 'r')
axs1[0].plot(dx, sweep_adj, linestyle='--', marker='s', label=r'$\mathcal{E}_{\tilde{h}_b}$', color = 'g')

axs1[0].set_xlabel(r'$\delta_x$', fontsize=22)
axs1[0].set_ylabel(r'$\mathcal{E}_{h_b}$', fontsize=25)

axs1[0].xaxis.set_major_formatter(FuncFormatter(gf.u_latex_sci_notation))

# $\mathcal{E}[h_b]$ = $\frac{\int \sqrt{(h_b - \hat{h_b})^2}\,dx}{\int \sqrt{(h_b)^2}\,dx}$
axs1[0].legend(loc = 'center left', fontsize=22)
axs1[0].tick_params(labelbottom=False)
axs1[0].set_xlabel('')

axs1[1].plot(dx, Usweep_pinn, linestyle='-', marker='o', label=r'$\mathcal{E}_{\hat{u}}$', color = 'r')
axs1[1].plot(dx, Usweep_adj, linestyle='--', marker='s', label=r'$\mathcal{E}_{\tilde{u}}$', color = 'g')

axs1[1].set_xlabel(r'$\delta_x$', fontsize=22)
axs1[1].set_ylabel(r'$\mathcal{E}_{u}$', fontsize=25)
# $\mathcal{E}[U]$ = $\frac{\int \sqrt{(u - \hat{u})^2}\,dx}{\int \sqrt{(u)^2}\,dx}$
axs1[1].legend(loc = 'center left', fontsize=22)
axs1[1].xaxis.set_major_formatter(FuncFormatter(gf.u_latex_sci_notation))

axs1[1].set_xticks(dx, labelsize=22)
axs1[1].set_xticklabels(dx_labels, fontsize=22)

labels = [r'$\mathrm{(a)}$', r'$\mathrm{(b)}$']
# labels = [r'$\text{(a)}$', r'$\text{(b)}$', r'$\text{(c)}$', r'$\text{(d)}$']

axs1[0].tick_params(which='both', direction='in', bottom=False, left=True , labelsize = 18)
axs1[-1].tick_params(which='both', direction='in', bottom=True, left=True , labelsize = 18)
axs1[-1].tick_params(axis='x' , which='both', direction='in', bottom=True, left=False , labelsize = 25)


for i, ax in enumerate(axs1):  # assuming axs3 is a 1D array of Axes
    ax.text(0.95, 0.90, labels[i], transform=ax.transAxes,
            ha='center', va='top', fontsize=18)

plt.tight_layout()
plt.savefig(f'{gf.figure_path}/normalized_total_error-hb_u.pdf')
