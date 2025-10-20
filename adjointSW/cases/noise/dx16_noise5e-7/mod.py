import json
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import curve_fit
import jax.numpy as jnp
import optax
from types import SimpleNamespace
import spooky as ps
from spooky.solvers.swhd_1d import SWHD_1D
from spooky.solvers.adjoint_swhd_1d import Adjoint_SWHD_1D


def check_dir(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory if it doesn't exist
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

def update_npy_file(npy_path, iit0):
    file = np.load(f'{npy_path}')
    storage = np.full(file.shape, np.nan, dtype=np.float64)
    storage[:iit0] = file[:iit0]
    return storage

def reset(fpm, bpm, fsolver, bsolver,
          hb=None,
          dg=None,
          dh0=None,
          du0=None):
    '''removes files if iit0 is set to 0. If not, restarts last run from last iit0'''
    if fpm.iit0 == 0:
        print('remove hbs content')
        for filename in os.listdir(fpm.hb_path):
            file_path = os.path.join(fpm.hb_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Remove the file
        print('remove forward content')
        for filename in os.listdir(fpm.out_path):
            file_path = os.path.join(fpm.out_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Remove the file
        print('remove backward content')
        for filename in os.listdir(bpm.out_path):
            file_path = os.path.join(bpm.out_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Remove the file
        if hb is None:
            hb = np.zeros_like(fsolver.true_hb)
        if dg is None:
            dg = np.zeros_like(fsolver.true_hb)
        if dh0 is None:
            dh0 = np.zeros_like(bsolver.true_hh0)
        if du0 is None:
            du0 = np.zeros_like(bsolver.true_uu0)
    else:
        fpm.iit = fpm.iit0
        bpm.iit = bpm.iit0
        # handle .dat files if restarting from a previous run

        bsolver.u_loss = update_npy_file(fpm.hb_path+'/u_loss.npy', bpm.iit0)
        bsolver.h_loss = update_npy_file(fpm.hb_path+'/h_loss.npy', bpm.iit0)
        bsolver.val = update_npy_file(fpm.hb_path+'/validation.npy', bpm.iit0)
        bsolver.hbs = update_npy_file(fpm.hb_path+'/hbs.npy', bpm.iit0)
        bsolver.dgs = update_npy_file(fpm.hb_path+'/dgs.npy', bpm.iit0)
        hb = bsolver.hbs[bpm.iit0-1]
        dg = bsolver.dgs[bpm.iit0-1]
    return hb, dg, dh0, du0

def early_stopping(w, loss, iit, patience=3):
    """
    Simple early stopping function.

    Args:
        w (int): Current patience counter.
        loss (list): List of loss values.
        patience (int): Number of steps to wait before stopping.

    Returns:
        (int, bool): Updated patience counter and stopping signal.
    """

    # Check if the loss improved
    print('loss compare: ' , loss[iit-1] , loss[iit-2])
    if loss[iit-1] <= loss[iit-2]:
        w = 0  # Reset patience
    else:
        w += 1

    # Stop if patience runs out
    if w >= patience:
        print(f"Stopping early after {w} iterations without improvement.")
        return w, True  # Signal to stop

    return w, False  # Keep going


def interpolate(mms, M, type='trig', knulls=None,domain=None , interp_domain=None):
    """
    Trigonometric interpolator via zero-padded FFT.

    Parameters
    ----------
    mms : array_like
        Input signal sampled at N points (assumed periodic).
    M : int
        Desired number of output points (M > N for upsampling).
    type : str
        Interpolation type ('trig' supported).

    Returns
    -------
    out : ndarray
        Interpolated signal at M points.
    """
    if type == 'trig':

        N = len(mms)
        # FFT of the original signal
        fft_mms = np.fft.fft(mms)
        if knulls!=None:
            for ki in knulls:
                fft_mms[ki]=0
            # Shift zero-frequency component to center
            fft_mms_shifted = np.fft.fftshift(fft_mms)

        # Zero-padding in frequency domain
        pad_width = (M - N) // 2
        if (M - N) % 2 == 0:
            padded_fft = np.pad(fft_mms_shifted, (pad_width, pad_width), mode='constant')
        else:
            padded_fft = np.pad(fft_mms_shifted, (pad_width, pad_width + 1), mode='constant')

        # Shift back and IFFT to get interpolated values
        padded_fft_unshifted = np.fft.ifftshift(padded_fft)
        interp = np.fft.ifft(padded_fft_unshifted) * (M / N)

        # Return real part (imaginary part should be ~0 due to Hermitian symmetry)
        return interp.real
    elif type == 'cubic_spline':
            # --- Cubic spline interpolation ---
        cs = CubicSpline(domain, mms)  # periodic if your Fourier case is periodic
        return cs(interp_domain)
    elif type == "quadratic":
        # --- Linear interpolation ---
        lin_interp = interp1d(domain, mms, kind='quadratic', fill_value="extrapolate")
        return lin_interp(interp_domain)
    elif type == "linear":
        # --- Linear interpolation ---
        lin_interp = interp1d(domain, mms, kind='linear', fill_value="extrapolate")
        return lin_interp(interp_domain)

# def gaussian(x, A, x0, sigma):
#    return A * np.exp(-(x - x0)**2 / (2 * sigma**2))

def gaussian_c2_with_offset(x, A, x0, c2, b):
    return b + A * np.exp(-((x - x0)**2) / (c2**2))

def interpolate_gaussian(x, y, x_new, fpm):
    """
    Fits a Gaussian curve to (x, y) and evaluates it at x_new.

    Parameters
    ----------
    x : array_like
        Original x values.
    y : array_like
        Original y values.
    x_new : array_like
        New x points where Gaussian is evaluated.

    Returns
    -------
    y_new : ndarray
        Interpolated y values  from fitted Gaussian.
    popt : tuple
        Fitted Gaussian parameters (A, x0, sigma).
    """


    # Robust initial guesses
    b_guess   = np.median(y)                   # baseline (≈ fpm.h0 if correct)
    A_guess   = float(y.max() - b_guess)       # amplitude
    x0_guess  = float(x[np.argmax(y)])         # peak location
    sigma_guess =  (np.max(x) - np.min(x)) / 4 # fpm.Lx/3 # 0.5

    popt, _ = curve_fit(gaussian_c2_with_offset, x, y, p0=[A_guess, x0_guess, sigma_guess, b_guess])

    return gaussian_c2_with_offset(x_new, *popt), popt


def plot_fields(fpm,
                hb,
                true_hb,
                out_u,
                true_u,
                out_h,
                true_h,
                uu0,
                true_uu0,
                hh0,
                true_hh0,
                noise_u=None,
                noise_h=None):
    f,axs = plt.subplots(ncols=3, figsize = (15,5))
    axs[0].plot(np.linspace(0,2*np.pi, len(hb)), hb , color = 'blue', label = '$\hat{h_b}$')
    axs[0].plot(np.linspace(0,2*np.pi, len(true_hb)), true_hb , alpha = 0.6, color = 'green', label = '$h_b$')
    axs[0].legend(fontsize=14)
    axs[0].set_xlabel('x', fontsize=12)
    axs[1].plot(np.linspace(0,2*np.pi, len(out_u)), out_u, color = 'blue', linestyle= '--', label = '$\hat{u}$')
    axs[1].plot(np.linspace(0,2*np.pi, len(true_u)), true_u, alpha = 1, color = 'green', label = '$u$')
    if fpm.noise:
        axs[1].plot(np.linspace(0,2*np.pi, len(noise_u)), noise_u, alpha = 0.5, color = 'red', label = '$u+\epsilon$')

    axs[1].legend(fontsize=14)
    axs[1].set_xlabel('x', fontsize=12)
    axs[2].plot(np.linspace(0,2*np.pi, len(out_h)), out_h, color = 'blue', linestyle= '--', label = '$\hat{h}$')
    axs[2].plot(np.linspace(0,2*np.pi, len(true_h)), true_h, alpha = 1, color = 'green', label = '$h$')
    if fpm.noise:
        axs[2].plot(np.linspace(0,2*np.pi, len(noise_h)), noise_h, alpha = 0.5, color = 'red', label = '$h+\epsilon$')

    axs[2].legend(fontsize=14)
    axs[2].set_xlabel('x', fontsize=12)
    plt.savefig(f'{fpm.hb_path}/fields.png')

    plt.figure()
    plt.plot(np.sqrt((hb-true_hb)**2) , label = '$\sqrt{(\hat{h_b}-h_b)^2}$')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('RSE', fontsize=12)
    plt.savefig(f'{fpm.hb_path}/hb_error.png')

    plt.figure()
    plt.plot(np.sqrt((uu0-true_uu0)**2) , label = '$\sqrt{(\hat{u_0}-u_0)^2}$')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('RSE', fontsize=12)
    plt.savefig(f'{fpm.hb_path}/u0_error.png')

    plt.figure()
    plt.plot(np.sqrt((hh0-true_hh0)**2) , label = '$\sqrt{(\hat{h_0}-h_0)^2}$')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('RSE', fontsize=12)
    plt.savefig(f'{fpm.hb_path}/h0_error.png')

    plt.figure(figsize=(15,5))
    plt.plot(np.linspace(0,2*np.pi, len(hb)) , hb ,label = '$\hat{h_b}$')
    plt.plot(np.linspace(0,2*np.pi, len(hb)) , true_hb ,label = '$h_b$')
    plt.xlabel('x', fontsize=12)
    plt.legend(fontsize=14)
    plt.savefig(f'{fpm.hb_path}/hb.png')

    plt.figure(figsize=(15,5))
    plt.plot(np.linspace(0,2*np.pi, len(uu0)) , uu0 ,label = '$\hat{u_0}$', color='green')
    plt.plot(np.linspace(0,2*np.pi, len(uu0)) , true_uu0, linestyle= '--' , alpha=0.6,label = '$u_0$', color='black')
    plt.xlabel('x', fontsize=12)
    plt.legend(fontsize=14)
    plt.savefig(f'{fpm.hb_path}/uu0.png')

    plt.figure(figsize=(15,5))
    plt.plot(np.linspace(0,2*np.pi, len(hh0)) , hh0 ,label = '$\hat{h_0}$', color='green')
    plt.plot(np.linspace(0,2*np.pi, len(hh0)) , true_hh0, linestyle= '--' , alpha=0.6,label = '$h_0$', color='black')
    plt.xlabel('x', fontsize=12)
    plt.legend(fontsize=14)
    plt.savefig(f'{fpm.hb_path}/hh0.png')

def plot_fourier(fpm, grid, hb,
                true_hb):
    Nx = fpm.Nx
    dx = grid.dx
    # Compute Fourier Transform of hb
    adj_hb_fft = np.fft.fft(hb)
    true_hb_fft = np.fft.fft(true_hb)

    hb_freq = np.fft.fftfreq(Nx, d=dx)

    # Take only the positive frequencies for plotting
    adj_hb_amplitude = np.abs(adj_hb_fft[:Nx // 2])
    true_hb_amplitude = np.abs(true_hb_fft[:Nx // 2])
    hb_freq_positive = hb_freq[:Nx // 2]

    # Plot the Fourier Amplitude Spectrum
    plt.figure(figsize=(10, 3))
    plt.loglog(hb_freq_positive,adj_hb_amplitude , color='green' , label = 'Spectrum of Adjoint $\hat{h_b}$')
    plt.loglog(hb_freq_positive,  true_hb_amplitude, color='black' ,alpha = 1 , label = 'Spectrum of $h_b$')
    plt.legend(fontsize=14)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid()
    plt.savefig(f'{fpm.hb_path}/fft_hb')


def plot_loss(fpm, u_loss, h_loss, val,  h0_loss=None, u0_loss=None):

    inverse_hb = fpm.inverse_hb
    inverse_u0 = fpm.inverse_u0
    inverse_h0 = fpm.inverse_h0

    if inverse_hb and inverse_u0 and inverse_h0:
        plt.figure()
        plt.semilogy(u_loss, label = '$(u-\hat{u})^2$')
        plt.semilogy(h_loss, label = '$(h- \hat{h})^2$')
        plt.xlabel('epochs', fontsize=12)
        plt.ylabel('Square Error', fontsize=12)
        plt.legend(fontsize=14)
        plt.savefig(f'{fpm.hb_path}/loss.png')

        f,axs = plt.subplots(nrows=3, figsize = (15,15))
        axs[0].semilogy(val, label = '$(h_b-\hat{h_b})^2$')
        axs[1].semilogy(u0_loss, label = '$(u_0-\hat{u_0})^2$')
        axs[-1].semilogy(h0_loss, label = '$(h_0-\hat{h_0})^2$')

        axs[-1].set_xlabel('epochs', fontsize=12)
        for ax in axs:
            ax.set_ylabel('Square Error', fontsize=12)
            ax.legend(fontsize=18)
        plt.savefig(f'{fpm.hb_path}/hb_u0_h0_val.png')

    elif inverse_hb and inverse_u0:
        plt.figure()
        plt.semilogy(u_loss, label = '$(u-\hat{u})^2$')
        plt.semilogy(h_loss, label = '$(h- \hat{h})^2$')
        plt.xlabel('epochs', fontsize=12)
        plt.ylabel('Square Error', fontsize=12)
        plt.legend(fontsize=14)
        plt.savefig(f'{fpm.hb_path}/loss.png')

        f,axs = plt.subplots(nrows=2, figsize = (15,15))
        axs[0].semilogy(val, label = '$(h_b-\hat{h_b})^2$')
        axs[1].semilogy(u0_loss, label = '$(u_0-\hat{u_0})^2$')

        axs[-1].set_xlabel('epochs', fontsize=12)
        for ax in axs:
            ax.set_ylabel('Square Error', fontsize=12)
            ax.legend(fontsize=18)
        plt.savefig(f'{fpm.hb_path}/hb_u0_val.png')

    if inverse_u0 and inverse_h0:
        plt.figure()
        plt.semilogy(u_loss, label = '$(u-\hat{u})^2$')
        plt.semilogy(h_loss, label = '$(h- \hat{h})^2$')
        plt.xlabel('epochs', fontsize=12)
        plt.ylabel('Square Error', fontsize=12)
        plt.legend(fontsize=14)
        plt.savefig(f'{fpm.hb_path}/loss.png')

        f,axs = plt.subplots(nrows=2, figsize = (15,15))
        axs[0].semilogy(u0_loss, label = '$(u_0-\hat{u_0})^2$')
        axs[-1].semilogy(h0_loss, label = '$(h_0-\hat{h_0})^2$')

        axs[-1].set_xlabel('epochs', fontsize=12)
        for ax in axs:
            ax.set_ylabel('Square Error', fontsize=12)
            ax.legend(fontsize=18)
        plt.savefig(f'{fpm.hb_path}/u0_h0_val.png')

    elif inverse_hb:
            plt.figure()
            plt.semilogy(u_loss, label = '$(u-\hat{u})^2$')
            plt.semilogy(h_loss, label = '$(h- \hat{h})^2$')
            plt.xlabel('epochs', fontsize=12)
            plt.ylabel('Square Error', fontsize=12)
            plt.legend(fontsize=14)
            plt.savefig(f'{fpm.hb_path}/loss.png')

            plt.figure()
            plt.semilogy(val, label = '$(h_b-\hat{h_b})^2$')
            plt.xlabel('epochs', fontsize=12)
            plt.ylabel('Square Error', fontsize=12)
            plt.legend(fontsize=14)
            plt.savefig(f'{fpm.hb_path}/hb_val.png')
    elif inverse_h0:
        plt.figure()
        plt.semilogy(u_loss, label = '$(u-\hat{u})^2$')
        plt.semilogy(h_loss, label = '$(h- \hat{h})^2$')
        plt.xlabel('epochs', fontsize=12)
        plt.ylabel('Square Error', fontsize=12)
        plt.legend(fontsize=14)
        plt.savefig(f'{fpm.hb_path}/loss.png')

        # f,axs = plt.subplots(nrows=1, figsize = (15,5))
        plt.figure()
        plt.semilogy(h0_loss, label = '$(h_0-\hat{h_0})^2$')

        plt.xlabel('epochs', fontsize=12)
        plt.ylabel('Square Error', fontsize=12)
        plt.legend(fontsize=18)
        plt.savefig(f'{fpm.hb_path}/h0_val.png')

def plot_dg(fpm,dg,DG):
    f,ax= plt.subplots(ncols=2, figsize = (15,5))
    ax[0].plot(dg , label = '$\int_{0}^{T} \partial_x h^{\dag} u \,dt$')
    ax[0].set_xlabel('x', fontsize=12)
    ax[0].legend(fontsize=14)
    ax[1].plot(DG , label = '$-$\eta$ lbfgs(\int_{0}^{T} \partial_x h^{\dag} u \,dt)$')
    ax[1].set_xlabel('x', fontsize=12)
    ax[1].legend(fontsize=14)
    plt.savefig(f'{fpm.hb_path}/dg.png')

def plot_du0(fpm,du0,DJ):
    f,ax= plt.subplots(ncols=2, figsize = (15,5))
    ax[0].plot(du0 , label = '$u^{\dag}(t=0)$')
    ax[0].set_xlabel('x', fontsize=12)
    ax[0].legend(fontsize=14)
    ax[1].plot(DJ , label = '-$\eta$ lbfgs('+'$u^{\dag}(t=0)$'+')')
    ax[1].set_xlabel('x', fontsize=12)
    ax[1].legend(fontsize=14)
    plt.savefig(f'{fpm.u0_path}/du0.png')

def plot_dh0(fpm,dh0,DH):
    f,ax= plt.subplots(ncols=2, figsize = (15,5))
    ax[0].plot(dh0 , label = '$h^{\dag}(t=0)$')
    ax[0].set_xlabel('x', fontsize=12)
    ax[0].legend(fontsize=14)
    ax[1].plot(DH , label = '-$\eta$ lbfgs('+'$h^{\dag}(t=0)$'+')')
    ax[1].set_xlabel('x', fontsize=12)
    ax[1].legend(fontsize=14)
    plt.savefig(f'{fpm.h0_path}/dh0.png')

def plot_hbs(fpm, true_hb, hbs):
    plt.figure()
    plt.plot(np.linspace(0,2*np.pi, len(hbs[0])), hbs[0] , color = 'blue', linestyle = '--', alpha = 0.7, label = '$\hat{h_b}$')
    for iteration in np.arange(0,fpm.iitN, fpm.ckpt):
        plt.plot(np.linspace(0,2*np.pi, len(hbs[iteration])), hbs[iteration] , color = 'blue', linestyle = '--', alpha = 0.7)

    plt.plot(np.linspace(0,2*np.pi, len(true_hb)), true_hb , alpha = 0.6, color = 'green', label = '$h_b$')
    plt.legend(fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylim((1-0.3)*true_hb.min() , (1+0.3)*true_hb.max())
    plt.savefig(f'{fpm.hb_path}/hbs.png')

def plot_u0s(fpm, true_u0, u0s):
    plt.figure()
    plt.plot(np.linspace(0,2*np.pi, len(u0s[0])), u0s[0] , color = 'blue', linestyle = '--', alpha = 0.7, label = '$\hat{u_0}$')
    for iteration in np.arange(0,fpm.iitN, fpm.ckpt):
        plt.plot(np.linspace(0,2*np.pi, len(u0s[iteration])), u0s[iteration] , color = 'blue', linestyle = '--', alpha = 0.7)

    plt.plot(np.linspace(0,2*np.pi, len(true_u0)), true_u0 , alpha = 0.6, color = 'green', label = '$u_0$')
    plt.legend(fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylim(true_u0.min()-0.3*true_u0.max() , (1+0.3)*true_u0.max())
    plt.savefig(f'{fpm.u0_path}/u0s.png')

def plot_h0s(fpm, true_h0, h0s):
    plt.figure()
    plt.plot(np.linspace(0,2*np.pi, len(h0s[0])), h0s[0] , color = 'blue', linestyle = '--', alpha = 0.7, label = '$\hat{h_0}$')
    for iteration in np.arange(0,fpm.iitN, fpm.ckpt):
        plt.plot(np.linspace(0,2*np.pi, len(h0s[iteration])), h0s[iteration] , color = 'blue', linestyle = '--', alpha = 0.7)

    plt.plot(np.linspace(0,2*np.pi, len(true_h0)), true_h0 , alpha = 0.6, color = 'green', label = '$h_0$')
    plt.legend(fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylim((1-0.0001)*true_h0.min() , (1+0.0001)*true_h0.max())
    plt.savefig(f'{fpm.h0_path}/h0s.png')

def plot_best_fields(pm, u0_loss, h0_loss, val, true_h0, h0s, true_u0, u0s, true_hb, hbs):

    # get minimum loss iit and plot its prediction
    if pm.inverse_u0:
        argmin_u0 = np.nanargmin(u0_loss)
        plt.figure()
        plt.plot(np.linspace(0,2*np.pi, len(u0s[argmin_u0])), u0s[argmin_u0] , color = 'blue', linestyle = '--', alpha = 0.7, label = '$\hat{u_0}$'+' epoch '+f'${argmin_u0}$')
        plt.plot(np.linspace(0,2*np.pi, len(true_u0)), true_u0 , alpha = 0.6, color = 'green', label = '$u_0$')
        plt.legend(fontsize=14)
        plt.xlabel('x', fontsize=12)
        plt.ylim(true_u0.min()-0.3*true_u0.max() , (1+0.3)*true_u0.max())
        plt.savefig(f'{pm.u0_path}/best_u0s.png')

    if pm.inverse_h0:
        argmin_h0 = np.nanargmin(h0_loss)
        plt.figure()
        plt.plot(np.linspace(0,2*np.pi, len(h0s[argmin_h0])), h0s[argmin_h0] , color = 'blue', linestyle = '--', alpha = 0.7, label = '$\hat{h_0}$'+' epoch '+f'${argmin_h0}$')
        plt.plot(np.linspace(0,2*np.pi, len(true_h0)), true_h0 , alpha = 0.6, color = 'green', label = '$h_0$')
        plt.legend(fontsize=14)
        plt.xlabel('x', fontsize=12)
        plt.ylim((1-0.0001)*true_h0.min() , (1+0.0001)*true_h0.max())
        plt.savefig(f'{pm.h0_path}/best_h0s.png')

    if pm.inverse_hb:
        argmin_hb = np.nanargmin(val)
        plt.figure()
        plt.plot(np.linspace(0,2*np.pi, len(hbs[argmin_hb])), hbs[argmin_hb] , color = 'blue', linestyle = '--', alpha = 0.7, label = '$\hat{h_b}$'+' epoch '+f'${argmin_hb}$')
        plt.plot(np.linspace(0,2*np.pi, len(true_hb)), true_hb , alpha = 0.6, color = 'green', label = '$h_b$')
        plt.legend(fontsize=14)
        plt.xlabel('x', fontsize=12)
        plt.ylim((1-0.001)*true_hb.min() , (1+0.001)*true_hb.max())
        plt.savefig(f'{pm.hb_path}/best_hbs.png')
