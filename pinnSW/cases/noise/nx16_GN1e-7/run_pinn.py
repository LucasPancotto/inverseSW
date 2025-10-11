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


# Get parameters
params = Run()

# NN params
layers = [2]+[params.hu]*params.layers+[2] # coords : t,x and output u, h

# Load data
X_data, Y_data, lambda_data, lambda_phys = generate_data(params, params.path)
lambda_phys = lambda_phys*params.lp

# Normalization layer
inorm = [X_data.min(0), X_data.max(0)]
means     = Y_data.mean(0)
# means[1] = params.h0
stds      = Y_data.std(0)
# stds[1]  = 0.05
onorm = [means, stds]
# Optimizer scheduler
if params.depochs:
    dsteps = params.depochs*len(X_data)/params.mbsize
    params.lr = keras.optimizers.schedules.ExponentialDecay(params.lr,
                                                            dsteps,
                                                            params.drate)
# Initialize model
from equations import SWHD as Eqs
eq_params = [np.float32(params.g) , np.float32(params.h0), np.float32(params.lm)]
PINN = PhysicsInformedNN(layers,
                         norm_in=inorm,
                         norm_out=onorm,
                         activation='siren',
                         optimizer=keras.optimizers.Adam(learning_rate=params.lr),
                         inverse=[{'type': 'func', 'layers': [2]+[params.hu_inv]*params.layers_inv+[1], 'mask': [0, 1]}])
PINN.optimizer.learning_rate.assign(params.lr)

# Validation function
PINN.validation = cte_validation(PINN, params, params.path, 80 , [0,1,50,100,150,170,-1])

# Train
PINN.train(X_data,
           Y_data,
           Eqs,
           epochs=params.epochs,
           batch_size=params.mbsize,
           alpha = 0.1,
           lambda_data=lambda_data,
           lambda_phys=lambda_phys,
           eq_params = eq_params,
           stds = stds,
           print_freq=100,
           valid_freq=100,
           save_freq=100,
           ckpt_folder_freq = 100,
           data_mask = [False, True]
           )
