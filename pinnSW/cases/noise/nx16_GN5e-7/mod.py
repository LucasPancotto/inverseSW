
from dom import *
import numpy as np

def generate_data(params, path):
    try:
        coords = np.load('data/coords.npy')
        fields = np.load('data/fields.npy')
        lambda_data = np.load('data/lambda_data')
        lambda_phys = np.load('data/lambda_phys')
    except:
        xx = 2*np.pi*np.linspace(0,1,num=params.N,endpoint=False) # N = 1024

        coords = []
        fields = []
        lambda_data = []
        lambda_phys = []

        # Load files
        uu = np.load(f'{path}/uums.npy', mmap_mode=None)
        hh = np.load(f'{path}/hhms.npy', mmap_mode=None) - params.h0 # h - h0
        if params.noise:
            uu = np.load(f'{path}/uums.npy', mmap_mode=None)
            hh = np.load(f'{path}/hhms.npy', mmap_mode=None) - params.h0 # h - h0
            hh = hh + np.random.normal(loc=0.0, scale=params.hhm_std, size=hh.shape)

        for ii in range(params.tsteps):
            vv = np.array([ff[ii].flatten() for ff in [uu,hh]])
            tt = ii*params.dt
            for jj in range(len(xx)):
                coords.append([tt, xx[jj]])
                fields.append([vv[0][jj], vv[1][jj]])
                lambda_phys.append(1.0)
                if jj%params.Nx:
                    lambda_data.append(0.0)
                else:
                    lambda_data.append(1.0)
            
        coords = np.array(coords).astype(np.float32)
        fields = np.array(fields).astype(np.float32)
        lambda_data = np.array(lambda_data).astype(np.float32)
        lambda_phys = np.array(lambda_phys).astype(np.float32)

        np.save('data/coords', coords)
        np.save('data/fields', fields)
        np.save('data/lambda_data', lambda_data)
        np.save('data/lambda_phys', lambda_phys)

    return coords, fields, lambda_data, lambda_phys

def plot_points(params, tidx):
    '''Creates a grid that matches the one in simulations'''
    N = params.N

    T = tidx*params.dt # a particular time in the simulation
    X = 2*np.pi*np.linspace(0,1,endpoint=False,num=params.N)

    T, X = np.meshgrid(T, X, indexing='ij')

    T = T.reshape(-1,1)
    X = X.reshape(-1,1)

    X = np.concatenate((T, X), 1)
    return X.astype(np.float32)

def cte_validation(self, params, path, tidx , tidxs):
    def validation(ep):
        N = params.N

        # Get predicted
        X_plot = plot_points(params, tidx=tidx)
        Y  = self.model(X_plot)[0].numpy() # model evaluated in all X,Y, T coordenates of data

        u_p = Y[:,0].reshape((N))
        th_p = Y[:,1].reshape((N)) + params.h0 # h - h0 + h0
        hb_p = self.model(X_plot)[1].numpy().reshape((N))

        pinn = np.array([u_p, th_p, hb_p])
        np.save(f"predicted.npy", pinn)
   
        # Load files
        uu = np.load(f'{path}/uums.npy', mmap_mode='r')
        hh = np.load(f'{path}/hhms.npy', mmap_mode='r')
        hb = np.load(f'{path}/hb.npy', mmap_mode='r')
        # ref = [abrirbin(f'{path}/{comp}.{tidx+1:03}.out', params.N)[:,512]
        #                 for comp in ['vx', 'vy', 'th', 'fs']]
        
        ref = [ff[tidx] for ff in [uu,hh] ]
        ref.append(hb)
        ref = np.array(ref)
        np.save(f"ref.npy", ref)

        err  = [np.sqrt(np.mean((ref[ff]-pinn[ff])**2))/np.std(ref[ff])
                if np.std(ref[ff]) !=0
                else np.sqrt(np.mean((ref[ff]-pinn[ff])**2))
                for ff in range(3)]
        # Loss functions
        output_file = open(self.dest + 'validation.dat', 'a')
        print(ep, *err,
            file=output_file)
        output_file.close() # print validation against simulation in validation.dat

        for t in tidxs:
            # Get predicted
            X_plot = plot_points(params, tidx=t)
            Y  = self.model(X_plot)[0].numpy() # model evaluated in all X,Y, T coordenates of data

            u_p = Y[:,0].reshape((N))
            th_p = Y[:,1].reshape((N)) + params.h0 
            hb_p = self.model(X_plot)[1].numpy().reshape((N))

            pinn = np.array([u_p, th_p, hb_p])
            np.save(f"predicted{t}.npy", pinn)

            ref = np.array(ref)
            ref = [ff[t] for ff in [uu,hh] ]
            ref.append(hb)
            ref = np.array(ref)
            np.save(f"ref{t}.npy", ref)

    return validation # returns the function validation(ep)

