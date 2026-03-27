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
import seaborn as sns
from types import SimpleNamespace

import pySPEC as ps
# from pySPEC.solvers import SWHD_1D, Adjoint_SWHD_1D

from utils import Getfiles
from matplotlib.ticker import FuncFormatter, ScalarFormatter

sns.set_style("white")
# palette = sns.color_palette("mako", as_cmap=True)
sns.set_palette(palette='Dark2')

figure_path = './figures'

nx = 1
last_iit = 96
interp = 'gauss'
multiple_run=False
plot_measurement = False

adjointpath = f'../adjointSW/cases/no_noise'+f'/dx{nx}'
gf = Getfiles(nx = nx,
                last_iit= last_iit,
                multiple_runs=multiple_run,
                plot_measurements=plot_measurement,
                normalized_data=True,
                adjointpath = adjointpath,
                figure_path=figure_path)
gf.get_paths()
gf.get_data()
gf.get_pinns()
gf.get_adjoints()
gf.plot_data()
gf.get_residuals()

tti = 2
'''residuals = [gf.mom_eq[tti], gf.mass_eq[tti], ]
labels = [r'$|\partial_t \hat{u} + \hat{u}\partial_x \hat{u} + g \partial_x \hat{h}|$',
 r'$|\partial_t \hat{h} + \partial_x [ \hat{u} (\hat{h} - \hat{h}_b)] |$' ]
colors = ['blue', 'orange']'''
residuals = [gf.mom_eq[tti]]
labels = [r'$|\partial_t \hat{u} + \hat{u}\partial_x \hat{u} + g \partial_x \hat{h}|$']
colors = ['blue']
# figure_path = 'no_noise_figures'
figure_path = './figures'

plt.close('all')
# f1,axs1 = plt.subplots(nrows = len(residuals), figsize=(15,10))
f1,axs1 = plt.subplots(nrows = len(residuals), figsize=(15,6))
axs1=[axs1]
# Force figure rendering so axis limits are accurate
f1.canvas.draw()

for i,pde in enumerate(residuals):

    axs1[i].plot(gf.domain, np.abs(pde), label = labels[i], color=colors[i], alpha=1)
    # axs1[i].set_title(r'$t=$'+f'${gf.tts_tau[tti]}$'+ r'$T$', fontsize=24, pad=10)
    axs1[i].yaxis.set_major_formatter(FuncFormatter(gf.u_latex_sci_notation))

    axs1[i].set_xlim(gf.domain[0], gf.domain[-1])  # <-- This removes x-axis margin
axs1[0].legend(fontsize=36, loc='upper right')
# axs1[1].legend(fontsize=36, loc='center right')
axs1[0].set_xlabel('$x/L$', fontsize = 24)

# Only show xticks and xlabel for the bottom row of axs
'''for ax in axs1[:-1]:
    ax.tick_params(labelbottom=False)
axs1[-1].set_xlabel(r'$x/L$', fontsize=24)'''


axs1[0].tick_params(which='both', direction='in', bottom=False, left=True, labelsize = 24)
# axs1[-1].tick_params(which='both', direction='in', bottom=True, left=True, labelsize = 24)
# labels = [r'$(a)$', r'$(b)$']
# labels = [r'$\text{(a)}$', r'$\text{(b)}$', r'$\text{(c)}$', r'$\text{(d)}$']
# labels = [r'$\mathrm{(a)}$', r'$\mathrm{(b)}$']

'''for i, ax in enumerate(axs1):  # assuming axs3 is a 1D array of Axes
    ax.text(0.95, 0.9, labels[i], transform=ax.transAxes,
            ha='center', va='top', fontsize=24)'''

f1.tight_layout()

f1.savefig(f'{gf.figure_path}/residuals_nx1_t05.pdf')
