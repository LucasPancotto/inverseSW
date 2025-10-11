# State preparation

import tensorflow as tf
from   tensorflow import keras
tf.keras.backend.set_floatx('float32')

from pinn      import PhysicsInformedNN

from   dom import *
from   mod import *
import numpy as np
import time
import random
import matplotlib.pyplot as plt


# Get parameters
params = Run()

# NN params
layers = [2]+[params.hu]*params.layers+[4] # coords : t,x,y , and output u , v , h

# Load data
X_data, Y_data = generate_data(params, '/share/data6/lpancotto/gauss_topo/gauss_topo011_interp_2/outs')

# Normalization layer
inorm = [X_data.min(0), X_data.max(0)]
means     = Y_data.mean(0)
means[2] = params.h0
stds      = Y_data.std(0)
stds[2]  = 0.01
onorm = [means, stds]

# Optimizer scheduler
if params.depochs:
    dsteps = params.depochs*len(X_data)/params.mbsize
    params.lr = keras.optimizers.schedules.ExponentialDecay(params.lr,
                                                            dsteps,
                                                            params.drate)
# Initialize model
from equations import SWHD as Eqs
eq_params = [np.float32(params.g) , np.float32(params.h0)]
PINN = PhysicsInformedNN(layers,
                         norm_in=inorm,
                         norm_out=onorm,
                         activation='siren',
                         optimizer=keras.optimizers.Adam(learning_rate=params.lr),
                         inverse=[{'type': 'func', 'layers': [2]+[params.hu_inv]*params.layers_inv+[1], 'mask': [0, 1]}],
                         eq_params=eq_params)
PINN.optimizer.learning_rate.assign(params.lr)

# Validation function
PINN.validation = cte_validation(PINN, params, '/share/data6/lpancotto/gauss_topo/gauss_topo011_interp_2/outs', 40)

tidx = 60
N = params.N
path =  '/share/data6/lpancotto/gauss_topo/gauss_topo011_interp_2/outs'
ref = [abrirbin(f'{path}/{comp}.{tidx+1:03}.out', params.N)[:,512]
               for comp in ['vx', 'vy', 'th', 'fs']]        
ref = np.array(ref)


T = tidx*params.dt # a particular time in the simulation
X = 2*np.pi*np.linspace(0,1,endpoint=False,num=params.N)

T, X = np.meshgrid(T, X, indexing='ij')

T = T.reshape(-1,1)
X = X.reshape(-1,1)
T = tf.convert_to_tensor(T)
X = tf.convert_to_tensor(X)

with tf.GradientTape(persistent=True) as tape:
    tape.watch([X,T])
    XX = tf.concat((T,X), 1)
    Y = PINN.model(XX)[0]
    u = Y[:,0]
    h = Y[:,2]
    b = Y[:,3]

    p = u*(h-b)

u_t = tf.cast(tape.gradient(u, T)[:,0], dtype='float32')
u_x = tf.cast(tape.gradient(u, X)[:,0], dtype='float32')
h_x = tf.cast(tape.gradient(h, X)[:,0], dtype='float32')
b_x = tf.cast(tape.gradient(b, X)[:,0], dtype='float32')
p_x = tf.cast(tape.gradient(p, X)[:,0], dtype='float32')
h_t = tf.cast(tape.gradient(h, T)[:,0], dtype='float32')

plt.figure(1)
plt.plot(X, h_t, label='h_t')
plt.plot(X, p_x, label='p_x')
plt.plot(X, h_t + p_x, label='res')
plt.title('mass eq')
plt.legend()

plt.figure(2)
plt.plot(X, u_t, label='u_t')
plt.plot(X, u*u_x, label='u*u_x')
# h_x = np.gradient(ref[2], 2*np.pi/N)
plt.plot(X, 2.0*h_x, label='g*h_x')
plt.plot(X, u_t + u*u_x + 2.0*h_x, label='res')
plt.title('momt eq')
plt.legend()

plt.figure(10)
plt.plot(X, ref[3], label='b_ref')
plt.plot(X, b, label='b_pinn')

plt.figure(30)
hxr = np.gradient(ref[3], 2*np.pi/N)
plt.plot(X, hxr, label='b_x_ref')
plt.plot(X, b_x, label='b_x_pinn')

plt.show()
