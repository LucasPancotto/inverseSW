#!/bin/bash
#SBATCH -J cos2d
#SBATCH -o %x.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00

ml gnu7/7.2.0
ml openmpi3
mpirun -np 1 ./SWHD
