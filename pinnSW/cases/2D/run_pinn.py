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

# Get path to data and time in which to conduct validation
path = params.path
val_time = params.val_time

# NN params
layers = [3]+[params.hu]*params.layers+[3] # coords : t,x,y , and output u , v , h
layers_inv = [3]+[params.hu_inv]*params.layers_inv+[1]
# Load data
X_data, Y_data = generate_data(params, path)
# Normalization layer
inorm = [X_data.min(0), X_data.max(0)]
means     = Y_data.mean(0)
# means[2] = params.h0
stds      = Y_data.std(0)
# stds[2]  = 0.01
onorm = [means, stds]

# save norms
np.save('inorm.npy', np.array(inorm) )
np.save('onorm.npy', np.array(onorm) )


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
                         inverse=[{'type': 'func', 'layers': layers_inv , 'mask': [0, 1, 1]}])
PINN.optimizer.learning_rate.assign(params.lr)

# Validation function
PINN.validation = cte_validation(PINN, params, path, val_time )

# Train
PINN.train(X_data,
           Y_data,
           Eqs,
           stds = stds,
           epochs=params.epochs,
           batch_size=params.mbsize,
           alpha = 0.1,
           lambda_data=1.0,
           lambda_phys=params.lp,
           eq_params = eq_params,
           print_freq=10,
           valid_freq=10,
           save_freq=10,
           ckpt_folder_freq = 10,
           verbose = True,
           data_mask = [False, False, True]
           )
