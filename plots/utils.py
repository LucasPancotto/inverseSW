import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter

from types import SimpleNamespace
import os

import pySPEC as ps
# from pySPEC.time_marching import SWHD_1D, Adjoint_SWHD_1D
plt.rcParams.update({"text.usetex": True,
    # "font.family": "serif",
    # "font.serif": ["Times"],
})

class Getfiles():
    """
    Gets files for plotting.
    """

    def __init__(self,
                 nx,
                 adjointpath=None,
                 pinn_path=None,
                 figure_path=None,
                 noise=False,
                 std_noise=None,
                 last_iit=None,
                 step = 80,
                 dt = 250,
                 tts = [0, 50, 100 , 150],
                 tau = 200,
                 history_iits=[2,5,10,20],
                 normalized_data=True,
                 L=2*np.pi/3,
                 multiple_runs=True,
                 plot_measurements=False,
                 plot_noise=False,
                 h_labels=True,
                 h_x_labels=True,
                 hb_labels=True,
                 hb_x_labels=True,
                 inverse_u0=False,
                 inverse_h0=False,
                 u0_path=None,
                 h0_path=None):

        self.param_path = '.'

        pm = json.load(open(f'{self.param_path}/params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
        pm.Lx = 2*np.pi*pm.Lx
        self.normalized_data = normalized_data
        self.L = L
        self.eta0 = None # will be catched from data
        self.h0 = pm.h0
        self.g = pm.g
        self.c = np.sqrt(self.g * self.h0)
        self.hb01 = None
        self.grid   = ps.Grid1D(pm)
        self.xx = self.grid.xx
        self.Nx = 1024
        self.Nl = L/(2*np.pi)*1024
        self.kl = self.Nx/self.Nl
        self.domain =  np.linspace(0,2*np.pi, 1024)
        if self.normalized_data:
            self.domain =  np.linspace(0,2*np.pi, 1024)/L
        self.plot_measurements = plot_measurements
        self.plot_noise = plot_noise
        self.h_labels = h_labels # auxiliary for masking labels in h plots to do multiplotting
        self.h_x_labels = h_x_labels
        self.hb_labels = hb_labels # auxiliary for masking labels in hb plots to do multiplotting
        self.hb_x_labels = hb_x_labels
        self.multiple_runs = multiple_runs
        self.pm = pm

        self.last_iit=last_iit
        self.nx = nx
        if self.normalized_data:
            self.kcut = (1024/nx)/self.kl
        else:
            self.kcut = 1024/nx
        self.tts = tts
        self.tts_tau = [tt/tau for tt in self.tts]
        if self.normalized_data:
            self.tau = tau/(self.L/self.c)
        else:
            self.tau = tau
        self.dt = dt
        self.val_step = step*dt
        self.noise = noise
        self.std_noise = std_noise
        self.inverse_u0 = inverse_u0
        self.inverse_h0 = inverse_h0
        self.u0_path = u0_path
        self.h0_path = h0_path
        self.u0 = None
        self.hh0 = None

        if adjointpath is not None:
            self.adjointpath = adjointpath
            if self.noise:
                self.pinn_path = f'../pinnsSW/cases/noise/nx{self.nx}_GN{self.std_noise}/'
            else:
                self.pinn_path = f'../pinnsSW/cases/no_noise/nx{self.nx}/'
            # self.pinn_path = pinn_path

            self.data_path = '../adjointSW/data/time_marching_swhd1D_DG-scaled-hbnoise/outs'
            self.hb_path = f'{self.adjointpath}/hbs'
            if self.inverse_u0:
                self.u0_path = f'{self.adjointpath}/u0s'
            if self.inverse_h0:
                self.h0_path = f'{self.adjointpath}/h0s'

        elif self.noise:
            self.pinn_path = f'../pinnsSW/cases/noise/nx{self.nx}_GN{self.std_noise}/'
            self.adjointpath = f'../adjointSW/cases/noise/'

            self.hb_path = f'{self.adjointpath}adjoint_lbfgs_sx{self.nx}_st250-{self.std_noise}/hbs'
            self.data_path = '../adjointSW/data/time_marching_swhd1D_DG-scaled-hbnoise/outs'
        else:
            self.pinn_path = f'../pinnsSW/cases/no_noise/nx{self.nx}/'
            self.adjointpath = f'../adjointSW/cases/no_noise/'

            self.hb_path = f'{self.adjointpath}adjoint_lbfgs_sx{self.nx}_st250/hbs'
            self.data_path = '../adjointSW/data/time_marching_swhd1D_DG-scaled-hbnoise/outs'

        if figure_path is None:
            self.figure_path = self.adjointpath+'figures'
        else:
            self.figure_path = figure_path
        self.pinn  = None
        self.pinn_times = None
        self.pinn_loss = None
        self.pinn_val = None
        self.uu  = None
        self.hh  = None
        self.hhms_noise=None
        self.history_iits=history_iits
        self.hbs_history=None
        self.hb = None
        self.pinn_uu_RSE=None
        self.adj_uu_RSE=None
        self.pinn_hb_RSE=None
        self.adj_hb_RSE=None
        self.pinn_uu_RSE_times=None
        self.adj_uu_RSE_times = None
        self.h_loss  = None
        self.u_loss  = None
        self.val  = None
        self.adj  = None
        self.uu_adj_times = None
        self.hh_adj_times = None

        self.uu_ref_times = None
        self.hh_ref_times  = None
        self.ref_noise  = None

        self.uu_pinns = None
        self.uu_stds = None
        self.hh_pinns = None
        self.hh_stds = None
        self.hb_pinn = None
        self.hb_std = None


        self.adj_hb_amplitude = None
        self.pinn_hb_amplitude = None
        self.std_pinn_hb_amplitude = None
        self.true_hb_amplitude = None

        self.adj_u_amplitude = None
        self.pinn_u_amplitude = None
        self.std_pinn_u_amplitude = None
        self.true_u_amplitude = None

        self.pinn_mse_hb_amplitude = None
        self.adj_mse_hb_amplitude = None
        self.pinn_mse_u_amplitude = None
        self.adj_mse_u_amplitude = None

        self.hb_freq_positive = None
        self.h_freq_positive_ = None


    def u_latex_sci_notation(self, x, _, int_=False):
        if x == 0:
            return r"$0$"
        exponent = int(np.floor(np.log10(abs(x))))
        base = x / 10**exponent
        if int_:
            base = int(x / 10**exponent)
            return r"${:d} \times 10^{{{:d}}}$".format(base, exponent)
        else:
            return r"${:.2f} \times 10^{{{:d}}}$".format(base, exponent) # return float base

    def h_latex_sci_notation(self, y, _, int_=False):
        if abs(y - self.h0) < 1e-12:
            return r"$0$"
        val = y - self.h0
        exponent = int(np.floor(np.log10(abs(val))))
        base = val / 10**exponent
        if int_:
            base = int(val / 10**exponent)
            return r"${:d} \times 10^{{{:d}}}$".format(base, exponent)
        else:
            return r"${:.2f} \times 10^{{{:d}}}$".format(base, exponent) # return float base

    def h_offset_latex_sci_notation(self, int_=False):
        if abs(self.h0) < 1e-12:
            return r"$0$"
        val = self.h0
        exponent = int(np.floor(np.log10(abs(val))))

        base = val / 10**exponent
        if int_:
            base = int(val / 10**exponent)
            return r"$\,+\, {:d} \times 10^{{{:d}}}$".format(base, exponent)
        else:
            return r"$\,+\, {:.2f} \times 10^{{{:d}}}$".format(base, exponent) # return float base

    def get_paths(self):
        if self.multiple_runs:
            self.pinn_paths = [self.pinn_path + r for r in os.listdir(self.pinn_path) if r.startswith('run')]
        else:
            self.pinn_paths = self.pinn_path

    def get_pinns(self, tpath=None):
        if self.multiple_runs:
            self.pinn = [np.load(f'{pinn_path}/{self.get_tpath(pinn_path, tpath=tpath)}/predicted.npy')  for pinn_path in self.pinn_paths]
            self.pinn_times = [[np.load(f'{pinn_path}/{self.get_tpath(pinn_path, tpath=tpath)}/predicted{tt}.npy') for tt in self.tts] for pinn_path in self.pinn_paths]
            self.pinn_loss = [np.loadtxt(f'{pinn_path}/{self.get_tpath(pinn_path, tpath=tpath)}/output.dat', unpack=True) for pinn_path in self.pinn_paths]
            self.pinn_val = [np.loadtxt(f'{pinn_path}/{self.get_tpath(pinn_path, tpath=tpath)}/validation.dat' , unpack=True) for pinn_path in self.pinn_paths]

            # mean of u,h pinns
            self.mean_pinns = np.array(self.pinn_times).mean(0) # axis 0 are the runs, axis 1 are tts, last axis are fields
            self.std_pinns = np.array(self.pinn_times).std(0)
            # mean of u,h,hb fields at validation time
            self.mean_pinn = np.array(self.pinn).mean(0)
            self.std_pinn = np.array(self.pinn).std(0)
        else:
            self.pinn = np.load(f'{self.pinn_path}/{self.get_tpath(self.pinn_path, tpath=tpath)}/predicted.npy')
            self.pinn_times = [np.load(f'{self.pinn_path}/{self.get_tpath(self.pinn_path, tpath=tpath)}/predicted{tt}.npy') for tt in self.tts]
            self.pinn_loss = np.loadtxt(f'{self.pinn_path}/{self.get_tpath(self.pinn_path, tpath=tpath)}/output.dat', unpack=True)
            self.pinn_val = np.loadtxt(f'{self.pinn_path}/{self.get_tpath(self.pinn_path, tpath=tpath)}/validation.dat' , unpack=True)

            self.mean_pinns = np.array(self.pinn_times) # axis 0 are tts, last axis are fields
            self.mean_pinn = np.array(self.pinn)
            self.std_pinns = np.zeros_like(self.pinn_times)
            self.std_pinn = np.zeros_like(self.pinn)



        if self.normalized_data:
            self.uu_pinns = [self.mean_pinns[i][0]/self.c for i in range(len(self.tts))]
            self.uu_pinn = self.mean_pinn[0]/self.c
            self.uu_stds = [self.std_pinns[i][0]/self.c for i in range(len(self.tts))]
            self.hh_pinns = [(self.mean_pinns[i][1]-self.h0)/self.eta0 for i in range(len(self.tts))]
            self.hh_pinn = (self.mean_pinn[1]-self.h0)/self.eta0
            self.hh_stds = [(self.std_pinns[i][1])/self.eta0 for i in range(len(self.tts))]

            self.hb_pinn = self.mean_pinn[-1]/self.h0
            self.hb_std = self.std_pinn[-1]/self.h0
        else:
            self.uu_pinns = [self.mean_pinns[i][0] for i in range(len(self.tts))]
            self.uu_pinn = self.mean_pinn[0]
            self.uu_stds = [self.std_pinns[i][0] for i in range(len(self.tts))]

            self.hh_pinns = [(self.mean_pinns[i][1]-self.h0) for i in range(len(self.tts))]
            self.hh_pinn = (self.mean_pinn[1]-self.h0)
            self.hh_stds = [(self.std_pinns[i][1]) for i in range(len(self.tts))]

            self.hb_pinn = self.mean_pinn[-1]
            self.hb_std = self.std_pinn[-1]

    def get_tpath(self, pinn_path, tpath=None):
        if tpath is None:
            try:
                return 't'+str(sorted([int(p.removeprefix('t')) for p in os.listdir(pinn_path) if p.startswith('t')])[-1])
            except:
                print(f'no tpath found in {pinn_path}')
                return None
        else:
            return tpath

    def get_adjoints(self):
        self.uu =  np.load(f'{self.hb_path}/uus.npy')  # all t uu
        self.hh =  np.load(f'{self.hb_path}/hhs.npy')  # all t hh
        hbs =  np.load(f'{self.hb_path}/hbs.npy')  # all iit hb
        if self.inverse_u0:
            if self.normalized_data:
                u0s =  np.load(f'{self.u0_path}/u0s.npy')/self.c  # all iit hb
            else:
                u0s =  np.load(f'{self.u0_path}/u0s.npy')
        if self.inverse_h0:
            if self.normalized_data:
                h0s =  np.load(f'{self.h0_path}/h0s.npy')/self.c  # all iit hb
            else:
                h0s =  np.load(f'{self.h0_path}/h0s.npy')
        if self.normalized_data:
            self.uu =  self.uu/self.c
            self.hh =  (self.hh-self.h0)/self.eta0
            hbs =  hbs/self.h0

        hbs = hbs[~np.isnan(hbs).any(axis=1)]  # Keep only rows without NaNs
        if self.last_iit is None:
            self.hb = hbs[-1]
            if self.inverse_u0:
                self.u0 = u0s[-1]
            if self.inverse_h0:
                self.hh0 = h0s[-1]
            self.last_iit = len(hbs)
        else:
            self.hb = hbs[self.last_iit-1]
            if self.inverse_u0:
                self.u0 = u0s[self.last_iit-1]
            if self.inverse_h0:
                self.hh0 = h0s[self.last_iit-1]
        print('last iit  '  , self.last_iit)
        self.hbs_history = [hbs[iit] for iit in self.history_iits]

        self.h_loss = np.load(f'{self.hb_path}/h_loss.npy')
        self.u_loss = np.load(f'{self.hb_path}/u_loss.npy')
        self.val = np.load(f'{self.hb_path}/validation.npy' )

        self.adj = [self.uu[self.val_step],self.hh[self.val_step],self.hb]
        self.uu_adj_times = [self.uu[val_step*250] for val_step in self.tts]
        self.hh_adj_times = [self.hh[val_step*250] for val_step in self.tts]

    def get_data(self):
        if self.normalized_data:
            true_uu =  np.load(f'{self.data_path}/uums.npy')/self.c  # all t true_uu normalized by propagation velocity
            true_eta =  np.load(f'{self.data_path}/hhms.npy') - self.h0 # all t true_hh - h0
            self.eta0 = true_eta[0].max()
            true_hh = true_eta/self.eta0 # normalize height field by eta0
            try:
                self.hhms_noise = (np.load(f'{self.hb_path}/hhms_noise.npy') - self.h0)/self.eta0  # all t hhm with noise
            except:
                print('no noise from data')
            true_hb = np.load(f'{self.data_path}/hb.npy') # true hb
            true_hb = true_hb/self.h0 # normalized by hb_0^(1)

            val_step = self.val_step

            self.ref = [true_uu[val_step],true_hh[val_step],true_hb]
            self.uu_ref_times = [true_uu[val_step*250] for val_step in self.tts]
            self.hh_ref_times = [true_hh[val_step*250] for val_step in self.tts]
        else:
            true_uu =  np.load(f'{self.data_path}/uums.npy')  # all t true_uu
            true_hh =  np.load(f'{self.data_path}/hhms.npy')  # all t true_hh
            try:
                self.hhms_noise = np.load(f'{self.hb_path}/hhms_noise.npy')  # all t hhm with noise
            except:
                print('no noise from data')
            true_hb = np.load(f'{self.data_path}/hb.npy') # true hb
            val_step = self.val_step

            self.ref = [true_uu[val_step],true_hh[val_step],true_hb]
            self.uu_ref_times = [true_uu[val_step*250] for val_step in self.tts]
            self.hh_ref_times = [true_hh[val_step*250] for val_step in self.tts]


    def get_integral_error(self):
        '''Calculates field errors in different, normalized by max true amplitude in that range'''
        d={'u_errors': None , 'hb_errors':None}

        # Epinn = self.integrate_ks(self.pinn_mse_hb_amplitude, self.hb_freq_positive_, kmin=0, kmax=512)/self.integrate_ks(self.true_hb_amplitude, self.hb_freq_positive_, kmin=0, kmax=512)
        # Eadj = self.integrate_ks(self.adj_mse_hb_amplitude, self.hb_freq_positive_, kmin=0, kmax=512)/self.integrate_ks(self.true_hb_amplitude, self.hb_freq_positive_, kmin=0, kmax=512)
        Epinn = self.integrate_x(self.pinn_hb_RSE, self.xx)/self.integrate_x(np.sqrt((self.hb)**2), self.xx)
        Eadj = self.integrate_x(self.adj_hb_RSE, self.xx)/self.integrate_x(np.sqrt((self.hb)**2), self.xx)
        # integrate error spectrum from k=0 to k=10 normalized by maximum amplitude of reference in those scales
        Epinn_0 = self.integrate_ks(self.pinn_mse_hb_amplitude, self.hb_freq_positive_, kmin=0, kmax=10)/self.integrate_ks(self.true_hb_amplitude,self.hb_freq_positive_, kmin=0, kmax=10)
        Eadj_0 = self.integrate_ks(self.adj_mse_hb_amplitude, self.hb_freq_positive_, kmin=0, kmax=10)/self.integrate_ks(self.true_hb_amplitude,self.hb_freq_positive_,kmin=0, kmax=10)

        # integrate error spectrum from k=40 to k=100
        Epinn_mid = self.integrate_ks(self.pinn_mse_hb_amplitude, self.hb_freq_positive_, kmin=40, kmax=100)/self.integrate_ks(self.true_hb_amplitude,self.hb_freq_positive_, kmin=40, kmax=100)
        Eadj_mid = self.integrate_ks(self.adj_mse_hb_amplitude, self.hb_freq_positive_, kmin=40, kmax=100)/self.integrate_ks(self.true_hb_amplitude,self.hb_freq_positive_, kmin=40, kmax=100)

        d['hb_errors']=[Epinn, Eadj, Epinn_0, Eadj_0, Epinn_mid, Eadj_mid]

        # Idem for u field:

        # Epinn = self.integrate_ks(self.pinn_mse_u_amplitude, self.h_freq_positive_, kmin=0, kmax=512)/self.integrate_ks(self.true_u_amplitude, self.h_freq_positive_, kmin=0, kmax=512)
        # Eadj = self.integrate_ks(self.adj_mse_u_amplitude, self.h_freq_positive_, kmin=0, kmax=512)/self.integrate_ks(self.true_u_amplitude, self.h_freq_positive_, kmin=0, kmax=512)
        pinn_uu_RSE_times_integrated = np.trapz(self.pinn_uu_RSE_times, self.tts, axis = 0)
        adj_uu_RSE_times_integrated = np.trapz(self.adj_uu_RSE_times, self.tts, axis = 0)
        uu_times_integrated = np.trapz(self.uu_ref_times, self.tts, axis = 0)

        Epinn = self.integrate_x(pinn_uu_RSE_times_integrated, self.xx)/self.integrate_x(np.sqrt((uu_times_integrated)**2), self.xx)
        Eadj = self.integrate_x(adj_uu_RSE_times_integrated, self.xx)/self.integrate_x(np.sqrt((uu_times_integrated)**2), self.xx)

        # integrate error spectrum from k=0 to k=10 normalized by maximum amplitude of reference in those scales
        Epinn_0 = self.integrate_ks(self.pinn_mse_u_amplitude, self.h_freq_positive_, kmin=0, kmax=10)/self.integrate_ks(self.true_u_amplitude,self.h_freq_positive_, kmin=0, kmax=10)
        Eadj_0 = self.integrate_ks(self.adj_mse_u_amplitude, self.h_freq_positive_, kmin=0, kmax=10)/self.integrate_ks(self.true_u_amplitude,self.h_freq_positive_,kmin=0, kmax=10)

        # integrate error spectrum from k=40 to k=100
        Epinn_mid = self.integrate_ks(self.pinn_mse_u_amplitude, self.h_freq_positive_, kmin=40, kmax=100)/self.integrate_ks(self.true_u_amplitude,self.h_freq_positive_, kmin=40, kmax=100)
        Eadj_mid = self.integrate_ks(self.adj_mse_u_amplitude, self.h_freq_positive_, kmin=40, kmax=100)/self.integrate_ks(self.true_u_amplitude,self.h_freq_positive_, kmin=40, kmax=100)
        d['u_errors']=[Epinn, Eadj, Epinn_0, Eadj_0, Epinn_mid, Eadj_mid]

        return d

    def integrate_ks(self, y, x, kmin, kmax):
        return np.trapz(y[kmin:kmax+1], x[kmin:kmax+1])

    def integrate_x(self, y, x):
        return np.trapz(y, x)

    def get_max_amp(self, y, kmin, kmax):
        amps = y[kmin:kmax+1]
        return amps.max()

    def integrate_max_amp(self, y, x, kmin, kmax):
        y_max = np.ones_like(y)*self.get_max_amp(y, kmin, kmax)
        return self.integrate_ks(y_max, x, kmin, kmax)

    def plot_losses(self):
        plt.close('all')
        plt.figure(0)
        plt.semilogy(self.h_loss, label=r'$\widetilde{(h- \hat{h})^2}$' )
        plt.xlabel('$iterations$' , fontsize=16)
        plt.ylabel('$\widetilde{(y - \hat{y})^2}$', fontsize=16)
        # plt.title('Adjoint Method Loss')
        plt.legend(fontsize=16)
        if self.noise:
            plt.savefig(f'{self.figure_path}/adj_h_loss_nx{self.nx}-{self.std_noise}.pdf')
        else:
            plt.savefig(f'{self.figure_path}/adj_h_loss_nx{self.nx}.pdf')

        plt.close('all')
        plt.figure(1)
        plt.semilogy(self.val, label = '$\widetilde{(h_b - \hat{h_b})^2}$' )
        plt.semilogy(self.u_loss , label = '$\widetilde{(u- \hat{u})^2}$'  )
        plt.xlabel('$iterations$', fontsize=16)
        plt.ylabel('$\widetilde{(y - \hat{y})^2}$', fontsize=16)
        # plt.title('Adjoint Method Validation')
        plt.legend( fontsize=16)
        if self.noise:
            plt.savefig(f'{self.figure_path}/adj_validation_nx{self.nx}-{self.std_noise}.pdf')
        else:
            plt.savefig(f'{self.figure_path}/adj_validation_nx{self.nx}.pdf')

        plt.close('all')
        plt.figure(2)
        if self.multiple_runs:
            run = self.pinn_loss[0]
            plt.semilogy(run[0,:], run[1,:], label='$L_d$', color = 'blue')
            plt.semilogy(run[0,:], run[2,:], label='$L_p$', color = 'green')
            for run in self.pinn_loss:
                plt.semilogy(run[0,:], run[1,:], color = 'blue')
                plt.semilogy(run[0,:], run[2,:], color = 'green')
        else:
            plt.semilogy(self.pinn_loss[0], self.pinn_loss[1], label='$L_d$')
            plt.semilogy(self.pinn_loss[0], self.pinn_loss[2], label='$L_p$')
        plt.xlabel('$epochs$', fontsize=16)
        plt.ylabel('$\widetilde{(y - \hat{y})^2}$', fontsize=16)
        # plt.title('PINN Loss')
        plt.legend( fontsize=16)
        if self.noise:
            plt.savefig(f'{self.figure_path}/pinn_loss_nx{self.nx}-{self.std_noise}.pdf')
        else:
            plt.savefig(f'{self.figure_path}/pinn_loss_nx{self.nx}.pdf')

        plt.close('all')
        plt.figure(3)
        if self.multiple_runs:
            run = self.pinn_val[0]
            plt.semilogy(run[0,:], run[1,:] , label = r'$\sqrt{\widetilde{(u- \hat{u})^2}}$' , color = 'blue' )
            plt.semilogy(run[0,:], run[-1,:] , label = r'$\sqrt{\widetilde{(h_b- \hat{h_b})^2}}$' , color = 'green' )
            for run in self.pinn_val:
                plt.semilogy(run[0,:], run[1,:], color = 'blue')
                plt.semilogy(run[0,:], run[-1,:], color = 'green')
        else:
            plt.semilogy(self.pinn_val[0], self.pinn_val[1] , label = r'$\sqrt{\widetilde{(u- \hat{u})^2}}$'  )
            plt.semilogy(self.pinn_val[0], self.pinn_val[-1] , label = r'$\sqrt{\widetilde{(h_b- \hat{h_b})^2}}$'  )
        plt.xlabel('$epochs$', fontsize=16)
        plt.ylabel(r'$\sqrt{\widetilde{(y - \hat{y})^2}}$', fontsize=16)
        # plt.title('PINN validation')
        plt.legend( fontsize=16)
        if self.noise:
            plt.savefig(f'{self.figure_path}/pinn_validation_nx{self.nx}-{self.std_noise}.pdf')
        else:
            plt.savefig(f'{self.figure_path}/pinn_validation_nx{self.nx}.pdf')

        plt.close('all')
        f,axs = plt.subplots(ncols=2, figsize= (15,3))
        if self.multiple_runs:
            run = self.pinn_loss[0]
            axs[0].semilogy(run[0,:], np.sqrt(run[1,:]), label='PINN $L_d$', color = 'blue')
            axs[0].semilogy(run[0,:], np.sqrt(run[2,:]), label='PINN $L_p$', color = 'green')
            for run in self.pinn_loss:
                axs[0].semilogy(run[0,:], np.sqrt(run[1,:]), color = 'blue')
                axs[0].semilogy(run[0,:], np.sqrt(run[2,:]), color = 'green')
        else:
            axs[0].semilogy(self.pinn_loss[0], np.sqrt(self.pinn_loss[1]), label='PINN $L_d$')
            axs[0].semilogy(self.pinn_loss[0], np.sqrt(self.pinn_loss[2]), label='PINN $L_p$')

        axs[0].set_xlabel('$epochs$', fontsize=16)
        axs[0].set_ylabel(r'$\sqrt{\widetilde{(y - \hat{y})^2}}$', fontsize=16)
        axs[0].legend( fontsize=16)

        axs[1].semilogy(np.sqrt(self.h_loss) , label = 'Adjoint Method $\sqrt{\widetilde{(h- \hat{h})^2}}$'  )
        axs[1].set_xlabel('$iterations$', fontsize=16)
        axs[1].set_ylabel('')
        axs[1].tick_params(labelleft=False)  # Also hides the y-axis tick labels
        axs[1].legend( fontsize=16)


        # Match y-axis limits
        y_min = min(axs[0].get_ylim()[0], axs[1].get_ylim()[0])
        y_max = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
        axs[0].set_ylim(y_min, y_max)
        axs[1].set_ylim(y_min, y_max)

        plt.tight_layout()
        if self.noise:
            plt.savefig(f'{self.figure_path}/pinn_adj_loss_nx{self.nx}-{self.std_noise}.pdf')
        else:
            plt.savefig(f'{self.figure_path}/pinn_adj_loss_nx{self.nx}.pdf')

        plt.close('all')
        f,axs = plt.subplots(ncols=2, figsize= (15,3))
        if self.multiple_runs:
            run = self.pinn_val[0]
            axs[0].semilogy(run[0,:], run[1,:] , label = 'PINN $\sqrt{\widetilde{(u- \hat{u})^2}}$' , color = 'blue' )
            axs[0].semilogy(run[0,:], run[-1,:] , label = 'PINN $\sqrt{\widetilde{(h_b- \hat{h_b})^2}}$'  , color = 'green')
            for run in self.pinn_val:
                axs[0].semilogy(run[0,:], run[1,:] , color = 'blue')
                axs[0].semilogy(run[0,:], run[-1,:] , color = 'green')
        else:
            axs[0].semilogy(self.pinn_val[0], self.pinn_val[1] , label = 'PINN $\sqrt{\widetilde{(u- \hat{u})^2}}$'  )
            axs[0].semilogy(self.pinn_val[0], self.pinn_val[-1] , label = 'PINN $\sqrt{\widetilde{(h_b- \hat{h_b})^2}}$'  )
        axs[0].set_xlabel('$epochs$', fontsize=16)
        axs[0].set_ylabel(r'$\sqrt{\widetilde{(y - \hat{y})^2}}$', fontsize=16)
        axs[0].legend( fontsize=16)

        axs[1].semilogy(np.sqrt(self.u_loss) , label = 'Adjoint Method $\sqrt{\widetilde{(u- \hat{u})^2}}$'  )
        axs[1].semilogy(np.sqrt(self.val), label = 'Adjoint Method $\sqrt{\widetilde{(h_b - \hat{h_b})^2}}$' )
        axs[1].set_xlabel('$iterations$', fontsize=16)
        axs[1].set_ylabel('')
        axs[1].tick_params(labelleft=False)  # Also hides the y-axis tick labels
        axs[1].legend( fontsize=16)

        # Match y-axis limits
        y_min = min(axs[0].get_ylim()[0], axs[1].get_ylim()[0])
        y_max = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
        axs[0].set_ylim(y_min, y_max)
        axs[1].set_ylim(y_min, y_max)

        plt.tight_layout()
        if self.noise:
            plt.savefig(f'{self.figure_path}/pinn_adj_val_nx{self.nx}-{self.std_noise}.pdf')
        else:
            plt.savefig(f'{self.figure_path}/pinn_adj_val_nx{self.nx}.pdf')

    def plot_fields(self, plot_all_pinns=False):
        plt.close('all')


        # Create 5 subplots in a row
        f, axs = plt.subplots(ncols=4, figsize=(12, 3))
        axs = axs.flatten()  # Ensure axs is a 1D array

        # Apply scientific notation formatting
        for ax in axs:
            # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.yaxis.set_major_formatter(FuncFormatter(self.u_latex_sci_notation))
            # ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
            # ax.ticklabel_format(style='plain', axis='y')
            ax.set_xlim(self.domain[0], self.domain[-1])  # <-- This removes x-axis margin



        # Loop over all subplots and plot data
        for i, ax in enumerate(axs):
            ax.plot(self.domain, self.uu_ref_times[i], label = r'$u$', color='black', alpha=0.7)
            ax.plot(self.domain, self.uu_adj_times[i], label = r'$\tilde{u}$' , color='green', alpha=0.5, linestyle = '--')

            # plot mean of runs
            fu = self.grid.forward(self.uu_pinns[i])
            std = self.uu_stds[i]
            fu[0] = 0
            ax.plot(self.domain, self.grid.inverse(fu), label = r'$\hat{u}$', color = 'red')
            if self.multiple_runs:
                ax.fill_between(self.domain, self.grid.inverse(fu) - std, self.grid.inverse(fu) + std, alpha=0.3) #, label= '$\hat{u}\pm$ $\sigma(\hat{u})$')

            ax.set_title(r'$\frac{t}{\tau}=$'+f'{self.tts_tau[i]}')
        axs[0].legend(fontsize=18)
        axs[0].set_xlabel('$x$', fontsize=18)
        axs[0].set_ylabel('$u$', fontsize=18)

        # Remove ylabel and y-tick labels from subplots except the first
        for i, ax in enumerate(axs):
            if i != 0:
                ax.set_ylabel('')
                ax.tick_params(labelleft=False)

        # Match y-axis limits across all subplots
        y_min = min(ax.get_ylim()[0] for ax in axs)
        y_max = max(ax.get_ylim()[1] for ax in axs)
        for ax in axs:
            ax.set_ylim(y_min, y_max)

        plt.tight_layout()
        if self.noise:
            plt.savefig(f'{self.figure_path}/u_nx{self.nx}-{self.std_noise}.pdf')
        else:
            plt.savefig(f'{self.figure_path}/u_nx{self.nx}.pdf')
        # plt.show

        # for field h
        # Create 5 subplots in a row
        f, axs = plt.subplots(ncols=4, figsize=(12, 3))
        axs = axs.flatten()  # Ensure axs is a 1D array

        # Apply scientific notation formatting
        for i, ax in enumerate(axs):
            # ax.yaxis.set_major_formatter(FuncFormatter(self.h_latex_sci_notation))

            # Enable LaTeX rendering
            plt.rcParams['text.usetex'] = True

            ax.set_xlim(self.domain[0], self.domain[-1])  # <-- This removes x-axis margin

                # Add your own offset label in the same place
            if i==0:
                ax.yaxis.set_major_formatter(FuncFormatter(self.h_latex_sci_notation))
                ax.text(
                    0.0, 1.02, self.h_offset_latex_sci_notation(),
                    transform=ax.transAxes,
                    fontsize=10,
                    va='bottom',
                    ha='left'
                ) # add offset only in the first frame

        # Loop over all subplots and plot data
        for i, ax in enumerate(axs):
            if self.plot_noise:
                ax.plot(self.domain, self.hhms_noise[i], label = r'$h+\epsilon$', color='blue', alpha=0.4)
            ax.plot(self.domain, self.hh_ref_times[i], label = r'$h$', color='black', alpha=0.7)
            ax.plot(self.domain, self.hh_adj_times[i], label = r'$\tilde{h}$' , color='green',  linestyle = '--', alpha=0.6)

            # plot mean of runs
            h = self.hh_pinns[i]
            std = self.hh_stds[i]
            ax.plot(self.domain, h, label = r'$\hat{h}$', color = 'red')
            if self.multiple_runs:
                ax.fill_between(self.domain, h - std, h + std, alpha=0.3)# , label= '$\hat{h}\pm$ $\sigma(\hat{h})$')

            if self.h_labels:
                ax.set_title(r'$\frac{t}{\tau}=$'+f'{self.tts_tau[i]}')
        if self.h_labels:
            axs[0].legend(fontsize=18)
        if self.h_x_labels:
            axs[0].set_xlabel('$x$', fontsize=18)
        else:
            for ax in axs:
                ax.tick_params(labelbottom=False)
                ax.set_xlabel('')
        axs[0].set_ylabel('$h$', fontsize=18)

        # Remove ylabel and y-tick labels from subplots except the first
        for i, ax in enumerate(axs):
            if i != 0:
                ax.set_ylabel('')
                ax.tick_params(labelleft=False)

        # Match y-axis limits across all subplots
        y_min = min(ax.get_ylim()[0] for ax in axs)
        y_max = max(ax.get_ylim()[1] for ax in axs)
        for ax in axs:
            ax.set_ylim(y_min, y_max)

        plt.tight_layout()
        if self.noise:
            plt.savefig(f'{self.figure_path}/h_nx{self.nx}-{self.std_noise}.pdf')
        else:
            plt.savefig(f'{self.figure_path}/h_nx{self.nx}.pdf')


        # plt.show

        plt.close('all')
        plt.figure(3, figsize=(15,5))
        plt.plot(self.domain,self.ref[-1] , color = 'black' , label = r'$h_b$', alpha = 1)
        plt.plot(self.domain, self.adj[-1] , label = r'$\tilde{h}_b$'  , color = 'green', linestyle = '--', alpha=0.6)
        plt.plot(self.domain, self.hb_pinn , label = r'$\hat{h}_b$', alpha = 1  , color = 'red')
        if self.multiple_runs:
            plt.fill_between(self.domain, self.hb_pinn - self.hb_std, self.hb_pinn + self.hb_std, alpha=0.3) # , label= '$\hat{h_b}\pm$ $\sigma(\hat{h_b})$')

        if self.hb_labels:
            plt.legend(fontsize=18)
        if self.hb_x_labels:
            plt.xlabel('$x$', fontsize=18)
        else:
            plt.tick_params(labelbottom=False)
            plt.xlabel('')

        if self.plot_measurements:
            if self.normalized_data:
                N = 1024
                x_min, x_max = 0, 2 * np.pi/self.L
                x = np.linspace(x_min, x_max, N, endpoint=False)
                stride = self.nx
                x_markers = x[::stride]
                y_loc = -0.01
                y_markers = np.zeros_like(x_markers)+y_loc
            else:
                N = 1024
                x_min, x_max = 0, 2 * np.pi
                x = np.linspace(x_min, x_max, N, endpoint=False)
                stride = self.nx
                x_markers = x[::stride]
                y_loc = -0.01
                y_markers = np.zeros_like(x_markers)+y_loc

            plt.plot(x_markers, y_markers, 'x', color='orange' )# , label='data point locations')
        plt.ylabel('$h_b$', fontsize=18)
        plt.xlim(self.domain[0], self.domain[-1])  # <-- This removes x-axis margin
        plt.tight_layout()
        if self.noise:
            plt.savefig(f'{self.figure_path}/hb_nx{self.nx}-{self.std_noise}.pdf')
        else:
            plt.savefig(f'{self.figure_path}/hb_nx{self.nx}.pdf')

        plt.close('all')
        plt.figure( figsize=(15,5))
        plt.plot(self.domain,self.ref[-1] , color = 'black' , label = '$h_b$', alpha = 1)
        plt.plot(self.domain, self.adj[-1] , label = r'$\tilde{h}_b$'  , color = 'green')
        for i,iit,color in zip([0,1,2,3], self.history_iits, ['red','orange','yellow','blue']):
            plt.plot(self.domain, self.hbs_history[i], label = f'Iteration {iit}', alpha = 1  , color = color, linestyle = '--')

        plt.ylabel('$h_b$', fontsize=18)
        plt.xlabel('$x$', fontsize=18)
        plt.legend(fontsize=18)
        plt.xlim(self.domain[0], self.domain[-1])  # <-- This removes x-axis margin
        plt.tight_layout()
        if self.noise:
            plt.savefig(f'{self.figure_path}/hb_adj_history_nx{self.nx}-{self.std_noise}.pdf')
        else:
            plt.savefig(f'{self.figure_path}/hb_adj_history_nx{self.nx}.pdf')

        # plt.show

    def plot_data(self):
        plt.close('all')
        plt.figure(3, figsize=(15,5))
        plt.plot(self.domain,self.ref[-1] , color = 'black' , label = '$h_b/h_0$', alpha = 1)
        # plt.ylabel('$h_b/h_0$', fontsize=18)
        plt.xlabel('$x/L$', fontsize=18)
        plt.legend(fontsize=18)
        plt.xlim(self.domain[0], self.domain[-1])  # <-- This removes x-axis margin
        plt.tick_params(which='both', direction='in', bottom=True, left=True, labelsize= 14)
        plt.tight_layout()
        plt.savefig(f'{self.figure_path}/hb.pdf')

    def get_fourier(self):
        dx = self.grid.dx

        # Compute Fourier Transform of hb
        adj_hb_fft = np.fft.fft(self.adj[-1])
        pinn_hb_fft = np.fft.fft(self.hb_pinn)
        true_hb_fft = np.fft.fft(self.ref[-1])
        hb_freq = np.fft.fftfreq(self.Nx, d=dx)
        # Take only the positive frequencies for plotting
        self.adj_hb_amplitude = self.pm.g * np.abs(adj_hb_fft[:self.Nx // 2])**2 / 2
        self.pinn_hb_amplitude = self.pm.g * np.abs(pinn_hb_fft[:self.Nx // 2])**2 / 2
        self.true_hb_amplitude = self.pm.g * np.abs(true_hb_fft[:self.Nx // 2])**2 / 2

        hb_freq_positive = hb_freq[:self.Nx // 2]
        if self.normalized_data:
            self.hb_freq_positive_ = hb_freq_positive*2*np.pi/self.kl
        else:
            self.hb_freq_positive_ = hb_freq_positive*2*np.pi

        # for u field
        adj_u_fft = np.fft.fft(self.adj[0])
        pinn_u_fft = np.fft.fft(self.uu_pinn)
        true_u_fft = np.fft.fft(self.ref[0])

        h_freq = np.fft.fftfreq(self.Nx, d=dx)

        # Take only the positive frequencies for plotting
        self.adj_u_amplitude = np.abs(adj_u_fft[:self.Nx // 2])**2 /2
        self.pinn_u_amplitude = np.abs(pinn_u_fft[:self.Nx // 2])**2 /2
        self.true_u_amplitude = np.abs(true_u_fft[:self.Nx // 2])**2 /2
        h_freq_positive = h_freq[:self.Nx // 2]
        if self.normalized_data:
            self.h_freq_positive_ = h_freq_positive*2*np.pi/self.kl
        else:
            self.h_freq_positive_ = h_freq_positive*2*np.pi


        # RSE for hb field, first real then fft
        pinn_hb_RSE =  np.sqrt(( self.hb_pinn - self.ref[-1] )**2)
        adj_hb_RSE =  np.sqrt(( self.adj[-1]- self.ref[-1] )**2)
        self.pinn_hb_RSE = pinn_hb_RSE
        self.adj_hb_RSE = adj_hb_RSE

        pinn_mse_fft_hb =  np.fft.fft(pinn_hb_RSE)
        adj_mse_fft_hb =  np.fft.fft(adj_hb_RSE)


        # Take only the positive frequencies for plotting
        self.pinn_mse_hb_amplitude = np.abs(pinn_mse_fft_hb[:self.Nx // 2])
        self.adj_mse_hb_amplitude = np.abs(adj_mse_fft_hb[:self.Nx // 2])

        # for u field
        true_u_fft = np.fft.fft(self.ref[0])

        # first filter mode 0 for pinn u
        fu = self.grid.forward(self.uu_pinn)
        fu[0]=0
        pinn_u_mean = self.grid.inverse(fu)
        # RSE for u field
        pinn_uu_RSE =  np.sqrt((pinn_u_mean - self.ref[0])**2)
        adj_uu_RSE = np.sqrt((self.adj[0] - self.ref[0])**2)
        self.pinn_uu_RSE=pinn_uu_RSE
        self.adj_uu_RSE=adj_uu_RSE

        # also calculate the total RSE over all times available for uu
        # self.pinn_times[runs,times,field, xx] (4, 5, 3, 1024)
        if self.multiple_runs:
            uu_pinn_mean_times = np.array(self.pinn_times)[:,:,0,:].mean(0) # mean over runs
            uu_adj_times = np.array(self.uu_adj_times)
            uu_ref_times = np.array(self.uu_ref_times)
            for i in range(uu_pinn_mean_times.shape[0]):
                # first filter mode 0 for pinn u
                fu = self.grid.forward(uu_pinn_mean_times[i,:])
                fu[0]=0
                uu_pinn_mean_times[i,:] = self.grid.inverse(fu)

            pinn_uu_RSE_times =  np.sqrt((uu_pinn_mean_times - uu_ref_times)**2)
            adj_uu_RSE_times = np.sqrt((uu_adj_times - uu_ref_times)**2)
            self.pinn_uu_RSE_times=pinn_uu_RSE_times
            self.adj_uu_RSE_times=adj_uu_RSE_times
        else:
            uu_pinn_mean_times = np.array(self.pinn_times)[:,0,:] # no runs
            uu_adj_times = np.array(self.uu_adj_times)
            uu_ref_times = np.array(self.uu_ref_times)
            for i in range(uu_pinn_mean_times.shape[0]):
                # first filter mode 0 for pinn u
                fu = self.grid.forward(uu_pinn_mean_times[i,:])
                fu[0]=0
                uu_pinn_mean_times[i,:] = self.grid.inverse(fu)

            pinn_uu_RSE_times =  np.sqrt((uu_pinn_mean_times - uu_ref_times)**2)
            adj_uu_RSE_times = np.sqrt((uu_adj_times - uu_ref_times)**2)
            self.pinn_uu_RSE_times=pinn_uu_RSE_times
            self.adj_uu_RSE_times=adj_uu_RSE_times

        # continue with fft for one time
        pinn_mse_fft_u =  np.fft.fft(pinn_uu_RSE)
        adj_mse_fft_u =  np.fft.fft(adj_uu_RSE)

        # Take only the positive frequencies for plotting
        self.pinn_mse_u_amplitude = np.abs(pinn_mse_fft_u[:self.Nx // 2])
        self.adj_mse_u_amplitude = np.abs(adj_mse_fft_u[:self.Nx // 2])



    def plot_fourier(self):
        plt.close('all')

        # Plot the Fourier Amplitude Spectrum
        plt.figure(figsize=(10, 3))
        plt.loglog(self.hb_freq_positive_,  self.true_hb_amplitude, color='black' ,alpha = 1 , label = r'$E_{h_b}$')
        plt.loglog(self.hb_freq_positive_,self.adj_hb_amplitude , color='green' , label = r'$E_{\tilde{h}_b}$',linestyle = '--',alpha=0.6)
        plt.loglog(self.hb_freq_positive_,self.pinn_hb_amplitude , color='red' , label = r'$E_{\hat{h}_b}$')

        if self.nx>1 and not self.noise:
            # plt.vlines(self.kcut, self.true_hb_amplitude.min(), self.true_hb_amplitude.max(), label = '$k = 1024/{nx}$', linestyle = '-.')
            kcut_index = np.where(self.hb_freq_positive_==self.kcut)[0][0]
            plt.annotate(
            '',
            xy=(self.kcut, self.adj_hb_amplitude[kcut_index]),        # Arrowhead position
            xytext=(self.kcut, 1e-3*self.adj_hb_amplitude[kcut_index]),    # Arrow tail position
            arrowprops=dict(arrowstyle='->', color='blue', linestyle='-.'),
            label='$k = 1024/{nx}$')
        if self.nx==1 or self.h_labels:
            plt.legend(fontsize=16) # only legend in first drawing
        if self.nx==128 or self.h_x_labels:
            plt.xlabel('$k$', fontsize=16) # only x axis label at last drawing
        else:
            plt.tick_params(labelbottom=False)
            plt.xlabel('')
        plt.ylabel('$E_{h_b}$', fontsize=16)
        if self.normalized_data:
            plt.xlim([1e0, 5e1])
        else:
            plt.xlim([1e0, 2e2])
        plt.tight_layout()
        if self.noise:
            plt.savefig(f'{self.figure_path}/fft_hb_nx{self.nx}-{self.std_noise}.pdf')
        else:
            plt.savefig(f'{self.figure_path}/fft_hb_nx{self.nx}.pdf')
        # Plot the Fourier Amplitude Spectrum
        plt.figure(figsize=(10, 3))
        plt.loglog(self.h_freq_positive_, self.true_u_amplitude, color='black' , alpha =1, label = r'$E_u$')
        plt.loglog(self.h_freq_positive_,self.adj_u_amplitude , color='green' , label = r'$E_{\tilde{u}}$', linestyle = '--',alpha=0.6)
        plt.loglog(self.h_freq_positive_,self.pinn_u_amplitude , color='red' , label = r'$E_{\hat{u}}$')

        if self.nx>1:
            kcut_index = np.where(self.hb_freq_positive_==self.kcut)[0][0]
            # plt.vlines(self.kcut, self.true_u_amplitude.max()/10000000000, self.true_u_amplitude.max(), label = '$k = 1024/{nx}$', linestyle = '-.')
            plt.annotate(
                        '',
                        xy=(self.kcut, self.true_u_amplitude[int(kcut_index)]),        # Arrowhead position
                        xytext=(self.kcut, 1e-14*self.true_u_amplitude[int(kcut_index)]),    # Arrow tail position
                        arrowprops=dict(arrowstyle='->', color='blue', linestyle='-.'),
                        label='$k = 1024/{nx}$')
        plt.legend(fontsize=16)
        plt.xlabel('$k$', fontsize=16)
        plt.ylabel('$E_u$', fontsize=16)
        if self.normalized_data:
            plt.xlim([1e0, 5e1])
        else:
            plt.xlim([1e0, 2e2])
        plt.tight_layout()
        if self.noise:
            plt.savefig(f'{self.figure_path}/fft_u_nx{self.nx}-{self.std_noise}.pdf')
        else:
            plt.savefig(f'{self.figure_path}/fft_u_nx{self.nx}.pdf')

    def plot_fourier_error(self):

        dx = self.grid.dx
        plt.close('all')

        # Plot the Fourier Amplitude Spectrum
        f,ax = plt.subplots(nrows = 1,figsize=(15, 4))
        ax.loglog(self.hb_freq_positive_,  self.true_hb_amplitude, color='black' ,alpha = 1 , label = '$E_{h_b}$')
        ax.loglog(self.hb_freq_positive_,self.adj_mse_hb_amplitude , color='green' , alpha = 0.7, label = 'adjoint $|\mathcal{F}[h_b-\hat{h_b}]|$',linestyle = '--')
        ax.loglog(self.hb_freq_positive_,self.pinn_mse_hb_amplitude , color='red' , alpha = 0.7, label = 'PINN $|\mathcal{F}[h_b-\hat{h_b}]|$')
        if self.nx<1:
            # ax.vlines(self.kcut, self.true_hb_amplitude.min(), self.true_hb_amplitude.max(), label = '$k = 1024/{self.nx}$', linestyle = '-.')
            plt.annotate(
                        '',
                        xy=(self.kcut, self.adj_mse_hb_amplitude[self.kcut]),        # Arrowhead position
                        xytext=(self.kcut, 1e-3*self.adj_mse_hb_amplitude[self.kcut]),    # Arrow tail position
                        arrowprops=dict(arrowstyle='->', color='blue', linestyle='-.'),
                        label='$k = 1024/{nx}$')
        # plt.loglog(hb_freq_positive,  self.true_hb_amplitude, color='black' ,alpha = 1 , label = 'Spectrum of $h_b$')
        ax.legend(fontsize=16)
        ax.set_xlabel('k', fontsize=16)
        ax.set_ylabel('$E_{h_b}$', fontsize=16)
        if self.normalized_data:
            ax.set_xlim([1e0/self.kl, 2e2/self.kl])
        else:
            ax.set_xlim([1e0, 2e2])
        plt.tight_layout()
        if self.noise:
            plt.savefig(f'{self.figure_path}/Error_fft_hb_nx{self.nx}-{self.std_noise}.pdf')
        else:
            plt.savefig(f'{self.figure_path}/Error_fft_hb_nx{self.nx}.pdf')

        f,ax = plt.subplots(nrows = 1,figsize=(15, 4))

        # Plot the Fourier Amplitude Spectrum
        ax.loglog(self.h_freq_positive_,  self.true_u_amplitude, color='black' ,alpha = 1 , label = '$E[u]$')
        ax.loglog(self.h_freq_positive_,self.pinn_mse_u_amplitude , color='red' , label = 'adjoint $|\mathcal{F}[u-\hat{u}]|$')
        ax.loglog(self.h_freq_positive_,self.adj_mse_u_amplitude , color='green' ,linestyle = '--', label = 'PINN $|\mathcal{F}[u-\hat{u}]|$')
        if self.nx<1:
            ax.vlines(self.kcut, self.true_u_amplitude.max()/(10**14), self.true_u_amplitude.max(), label = '$k = 1024/{nx}$', linestyle = '-.')

        ax.legend(fontsize=16)
        ax.set_xlabel('k', fontsize=16)
        ax.set_ylabel('$E[u]$',  fontsize=16)
        ax.set_xlim([1e0, 2e2])
        plt.tight_layout()
        if self.noise:
            plt.savefig(f'{self.figure_path}/Error_fft_u_nx{self.nx}-{self.std_noise}.pdf')
        else:
            plt.savefig(f'{self.figure_path}/Error_fft_u_nx{self.nx}.pdf')
