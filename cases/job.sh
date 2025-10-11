#!/bin/bash

#SBATCH -J hb_h0_u0_nf_gauss_dx1-dt1
#SBATCH -N 1
#SBATCH -o error.out
#SBATCH -n 1
#SBATCH -t 48:00:00
 
ml cuda/11.8
ml python/3.11.3
# Activate the virtual environment
source /share/data6/lpancotto/pyenv/bin/activate
python3 adjoint_GD.py
