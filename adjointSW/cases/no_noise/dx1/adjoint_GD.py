import json
import numpy as np
import matplotlib.pyplot as plt
import os
from types import SimpleNamespace
# jax for optimization tools
import jax
jax.config.update("jax_enable_x64", True)
import jax.random as jrd
import optax
# import spooky tools and solvers
import spooky as ps
from spooky.solvers.swhd_1d import SWHD_1D
from spooky.solvers.adjoint_swhd_1d import Adjoint_SWHD_1D
# import utility functions
from mod import *

current_dir = os.path.dirname(os.path.abspath(__file__))
# Parse JSON into an object with attributes corresponding to dict keys for forward solver
pm = json.load(open(f'{current_dir}/params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
fpm = json.load(open(f'{current_dir}/params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
# scale domain to [0, 2pi]
fpm.Lx = 2*np.pi*fpm.Lx
# path for outputs
fpm.out_path = fpm.forward_out_path
check_dir(fpm.out_path) # make out path if it doesn't exist
# forward time step
fpm.ostep = fpm.forward_ostep

# Parse JSON into an object with attributes corresponding to dict keys for backward solver
bpm = json.load(open(f'{current_dir}/params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
# scale domain to [0, 2pi]
bpm.Lx = 2*np.pi*bpm.Lx
# path for outputs
bpm.out_path = bpm.backward_out_path
check_dir(bpm.out_path) # make out path if it doesn't exist
# backward time step
bpm.ostep = bpm.backward_ostep

# remove all files from result paths if restarting GD
check_dir(bpm.hb_path)
check_dir(bpm.u0_path)
check_dir(bpm.h0_path)

# Initialize grid
grid   = ps.Grid1D(fpm)
fsolver = SWHD_1D(fpm)
# backward solver inherits attributes from forward solver
bsolver = Adjoint_SWHD_1D(bpm, fsolver)
# get measurments for adjoint solver
bsolver.get_measurements()
# total number of iterations for forward-backward loop
total_iterations = bpm.iitN - bpm.iit0 - 1

# True velocity initial conditions, for reference
v1 = 0.00025
v2 =  0.5
v3 = 2
true_uu0 = v1 * np.exp(-((grid.xx - np.pi/v3) ** 2) / v2 ** 2)

# True height initial conditions, for reference
c1 = 5e-5
c2 = 0.5
c3 = 2
true_hh0 = fpm.h0 + c1 * np.exp(-((grid.xx - np.pi/c3) ** 2) / c2 ** 2)
hh0 = true_hh0

# build ansatz for initial conditions from sparse measurements
if pm.hh0_interp=='true':
    hh0 = true_hh0
elif pm.hh0_interp=='trig':
    hh0 = interpolate(bsolver.hhms_sparse[0,::pm.sx], M=len(grid.xx,), knulls=[-1,-2,-3,-4,-5,-6])
elif pm.hh0_interp=='quadratic':
    hh0 = interpolate(bsolver.hhms_sparse[0,::pm.sx], M=len(grid.xx,), type='quadratic', domain=grid.xx[::pm.sx] , interp_domain=grid.xx)
elif pm.hh0_interp=='cubic_spline':
    hh0 = interpolate(bsolver.hhms_sparse[0,::pm.sx], M=len(grid.xx,), type='cubic_spline', domain=grid.xx[::pm.sx] , interp_domain=grid.xx)
elif pm.hh0_interp=='linear':
    hh0 = interpolate(bsolver.hhms_sparse[0,::pm.sx], M=len(grid.xx,), type='linear', domain=grid.xx[::pm.sx] , interp_domain=grid.xx)
elif pm.hh0_interp=='flat':
    hh0 = pm.h0*np.ones_like(true_hh0)
elif pm.hh0_interp=='simple':
    hh0 = fpm.h0 + 0.95*c1 * np.exp(-((grid.xx - np.pi/c3) ** 2) / c2 ** 2)
elif pm.hh0_interp=='gauss':
    hh0 = interpolate_gaussian(x=grid.xx[::pm.sx], y=bsolver.hhms_sparse[0,::pm.sx], x_new=grid.xx, fpm=fpm)[0]
# use initial condition ansatz to build initial velocity condition from linearized SW solution
uu0 = (hh0-pm.h0)*np.sqrt(pm.g/pm.h0)
# plot initial condition measurements and ansatz
plt.close('all')
plt.scatter(grid.xx[::pm.sx], bsolver.hhms_sparse[0,::pm.sx], label='$h_0$')
plt.plot(grid.xx, hh0,label='$h_0$ interpolation')
plt.legend()
plt.savefig(f'{pm.h0_path}/initialh_interp.png')
plt.close('all')
plt.scatter(grid.xx[::pm.sx], true_uu0[::pm.sx], label='$u_0$')
plt.plot(grid.xx, uu0,label=r'$\tilde{u}_0=\sqrt{\frac{g}{h_0}}$')
plt.legend()
plt.savefig(f'{pm.u0_path}/initialu_interp.png')

# update true fields in forward and backward solver
fsolver.update_true_hb() # update true hb from data archive
bsolver.update_true_hb()
bsolver.update_true_uu0(true_uu0)
bsolver.update_true_hh0(true_hh0)

# reset gradients, hb ansatz and directories
hb, dg, dh0, du0 = reset(fpm, bpm, fsolver, bsolver) # flat hb ansatz
# update initial ansatz and gradients in forward and backward solver
fsolver.update_hb(hb)
bsolver.update_hb(hb)
bsolver.update_dg(dg)
bsolver.update_du0(du0)
bsolver.update_dh0(dh0)
bsolver.update_uu0(uu0)
bsolver.update_hh0(hh0)

# update fields and initialize hbs history in backward solver
bsolver.update_fields(fsolver)
bsolver.update_hbs(pm.iit0)
bsolver.update_dgs(pm.iit0)
bsolver.update_uu0s(pm.iit0)
bsolver.update_du0s(pm.iit0)
bsolver.update_hh0s(pm.iit0)
bsolver.update_dh0s(pm.iit0)

# Optimizer definitions
# choose gradient descent optimizer
# if pm.optimizer == 'lbfgs':
# Define objective
dim = 1024
mat = jrd.normal(jrd.PRNGKey(0), (dim, dim))
mat = mat @ mat.T  # Ensure mat is positive semi-definite

# learning rate for lbfgs
lgd = bpm.lgd
u0lgd = bpm.u0lgd
h0lgd = bpm.h0lgd

# lbfgs initialization
opt = optax.scale_by_lbfgs()
u0_opt = optax.scale_by_lbfgs()
h0_opt = optax.scale_by_lbfgs()
# Initialize optimization
state = opt.init(hb)
u0_state = opt.init(uu0)
h0_state = opt.init(hh0)

# Forward-backward loop
for iit in range(fpm.iit0 + 1, fpm.iitN):
    # update iit
    fpm.iit = iit
    bpm.iit = iit

    # catch initial conditions for foward integration
    uu = uu0
    hh = hh0
    fields = [uu, hh]
    # Forward integration
    print(f'iit {iit} : evolving forward')
    fsolver.evolve(fields, fpm.T, bstep=fpm.bstep, ostep=fpm.ostep)

    # update fields for backward integration
    bsolver.update_fields(fsolver)
    # Null initial conditions for adjoint state
    uu_ = np.zeros_like(grid.xx)
    hh_ = np.zeros_like(grid.xx)
    fields = [uu_, hh_]
    # update measurements (forcing terms) for backward solver
    bsolver.get_sparse_forcing()
    # Backward integration
    print(f'\niit {iit} : evolving backward')
    fields = bsolver.evolve(fields, bpm.T, bstep=bpm.bstep, ostep=bpm.ostep)

    # if true, optimize for hb
    if pm.inverse_hb:
        # integrate h_*ux from T to t=0
        print(f'\niit {iit} : calculate dg/dhb')
        Nt = round(fpm.T/fpm.dt)
        dg = np.trapz( bsolver.hx_uu, dx = 1e-4, axis = 0)

        # update hb values
        print(f'\niit {iit} : update hb')
        DG, state = opt.update(dg, state, hb)
        hb = hb - bpm.lgd * DG
        # update hb in forward solver
        fsolver.update_hb(hb)
        # update hb and dg in backward solver
        bsolver.update_hb(hb)
        bsolver.update_dg(dg)

    # if true, optimize for u0
    if pm.inverse_u0:
        # update u0
        print(f'\niit {iit} : getting dg/du0')
        du0 = bsolver.uus_[-1] # dg/du0 = adjoint_u(t=0)
        # update u0 values
        print(f'\niit {iit} : update u0')
        DJ, u0_state = u0_opt.update(du0, u0_state, uu0)
        uu0 = uu0 - bpm.u0lgd * DJ
        # update uu0 and du0 in backward solver
        bsolver.update_uu0(uu0)
        bsolver.update_du0(du0)

    # if true, optimize for h0
    if pm.inverse_h0:
        # update u0
        print(f'\niit {iit} : getting dg/dh0')
        dh0 = bsolver.hhs_[-1] # dg/du0 = adjoint_u(t=0)
        # update u0 values
        print(f'\niit {iit} : update h0')
        DH, h0_state = h0_opt.update(dh0, h0_state, hh0)
        hh0 = hh0 - bpm.h0lgd * DH
        # update hh0 and dh0 in backward solver
        bsolver.update_hh0(hh0)
        bsolver.update_dh0(dh0)
    print(f'\niit {iit} : saved new fields')

    # update loss
    bsolver.update_loss(iit-1)
    bsolver.update_val(iit-1)

    # Early stopping
    if pm.early_stop:
        if iit <=2:
            w, stop = [0,0]
        else:
            w, stop = early_stopping(w, bsolver.val, iit, patience=3)

        if stop:
            print(f"Early stopping triggered at step {iit}")
            break

    if iit%pm.ckpt==0:
        print(f'\n checkpoint at iit = {iit}')
        # update field history
        bsolver.update_uu0s(iit)
        bsolver.update_hh0s(iit)
        bsolver.update_hbs(iit)
        if pm.save_gradients:
            bsolver.update_dgs(iit)
            bsolver.update_du0s(iit)
            bsolver.update_dh0s(iit)
        # Plot fields
        print(f'\niit {iit} : plot')
        tval = int(fpm.T/fpm.dt*0.5)
        out_u = bsolver.uus[tval]
        out_h = bsolver.hhs[tval]
        plt.close("all")
        if pm.noise: # if measurements with noise
            true_u = bsolver.uums_[tval]
            true_h = bsolver.hhms_[tval]
            noise_u = bsolver.uums[tval]
            noise_h = bsolver.hhms[tval]
            plot_fields(fpm,
                        hb,
                        fsolver.true_hb,
                        out_u,
                        true_u,
                        out_h,
                        true_h,
                        noise_u,
                        noise_h)
        else: # if measurements without noise
            true_u = bsolver.uums[tval]
            true_h = bsolver.hhms[tval]
            plot_fields(fpm,
            hb,
            fsolver.true_hb,
            out_u,
            true_u,
            out_h,
            true_h,
            uu0,
            true_uu0,
            hh0,
            true_hh0)

        # plot gradients
        if pm.inverse_hb:
                plot_dg(fpm,dg,- bpm.lgd * DG)
        if pm.inverse_u0:
            plot_du0(fpm,du0, DJ =  -bpm.u0lgd * DJ)
        if pm.inverse_h0:
            plot_dh0(fpm,dh0, DH =  -bpm.h0lgd * DH)
        # plot hb, u0, h0 histories
        plot_hbs(fpm, fsolver.true_hb, bsolver.hbs)
        plot_u0s(fpm, true_uu0, bsolver.uu0s)
        plot_h0s(fpm, true_hh0, bsolver.hh0s)
        # plot loss history
        plot_loss(fpm, bsolver.u_loss, bsolver.h_loss, val=bsolver.val, h0_loss=bsolver.h0_loss, u0_loss=bsolver.u0_loss)

        # plot lowest error predictions
        plot_best_fields(pm, u0_loss=bsolver.u0_loss,
        h0_loss=bsolver.h0_loss,
        val=bsolver.val,
        true_h0=true_hh0,
        h0s=bsolver.hh0s,
        true_u0=true_uu0,
        u0s=bsolver.uu0s,
        true_hb=bsolver.true_hb,
        hbs=bsolver.hbs)

        # save losses and histories
        np.save(f'{fpm.hb_path}/u_loss.npy', bsolver.u_loss)
        np.save(f'{fpm.hb_path}/h_loss.npy', bsolver.h_loss)
        if pm.inverse_u0:
            np.save(f'{fpm.u0_path}/u0_loss.npy', bsolver.u0_loss)
            np.save(f'{fpm.u0_path}/u0s.npy', bsolver.uu0s)
        if pm.inverse_h0:
            np.save(f'{fpm.h0_path}/h0_loss.npy', bsolver.h0_loss)
            np.save(f'{fpm.h0_path}/h0s.npy', bsolver.hh0s)
        if pm.inverse_hb:
            np.save(f'{fpm.hb_path}/validation.npy', bsolver.val)
            np.save( f'{fpm.hb_path}/hbs.npy', bsolver.hbs)
        np.save( f'{fpm.hb_path}/uus.npy', bsolver.uus)
        np.save( f'{fpm.hb_path}/hhs.npy', bsolver.hhs)
        np.save( f'{fpm.hb_path}/dgs.npy', bsolver.dgs)
    # done
    print(f'done iit {fpm.iit}')
