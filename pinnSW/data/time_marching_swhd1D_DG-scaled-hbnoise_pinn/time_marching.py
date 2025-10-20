'''
Pseudo-spectral solver for the 1D SWHD equation
'''

import json
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import os
import sys
import spooky as ps
from spooky.solvers.swhd_1d import SWHD_1D

current_dir = os.path.dirname(os.path.abspath(__file__))
# Parse JSON into an object with attributes corresponding to dict keys.
pm = json.load(open(f'{current_dir}/params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
pm.Lx = 2*np.pi*pm.Lx

parent_dir = os.path.abspath(os.path.join(current_dir, '..'))  # One directory up
sys.path.insert(0, parent_dir)  # Add it to the *front* of the import path

from field_generator import Generator  # Now you can import safely

# Initialize solver
grid   = ps.Grid1D(pm)
solver = SWHD_1D(pm)

gen = Generator(current_dir, hb_noise=True, uh_noise=False)
# create noisy u, h
uu,hh = gen.get_uh()
# create noisey hb
hb = gen.get_hb()

solver.update_hb(hb)

np.save(f'{pm.out_path}/hb.npy', hb) # the fixed, true hb for adjoint loop later
fields = [uu, hh]

# Evolve
fields = solver.evolve(fields, pm.T, bstep=pm.bstep, ostep=pm.ostep)

# Plot Balance
bal = np.loadtxt(f'{pm.out_path}/balance.dat', unpack=True)
# Plot fields
tval = int(pm.T/(pm.dt*pm.ostep)*0.5)
out_u = np.load(f'{pm.out_path}/uums.npy')[tval] # all uu fields in time tval
out_h = np.load(f'{pm.out_path}/hhms.npy')[tval] # all uu fields in time tval
out_hb = np.load(f'{pm.out_path}/hb.npy')

f,axs = plt.subplots(ncols=3, figsize = (15,5))

axs[0].plot(out_hb , label = 'hb')
axs[0].legend()
axs[1].plot(out_u , label = 'u')
axs[1].legend()
axs[2].plot(out_h , label = 'h')
axs[2].legend()
plt.savefig(f'{pm.out_path}/fields.png')
