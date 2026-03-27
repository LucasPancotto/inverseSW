Directory `adjointSW` contains code for Adjoint State method cases, and directory `pinnSW` contains code for PINN cases. Both directories work independently.

For each method, the code to generate data points is contained in `adjointSW/data/time_marching_swhd1D_DG-scaled-hbnoise` and `pinnSW/data/time_marching_swhd1D_DG-scaled-hbnoise_pinn` respectively. The cases for each method are contained in  `adjointSW/cases` and `pinnSW/cases`, respectively.

Each directory `cases/` contains:
- A `no_noise/` folder, that contains several subdirectories which are cases with different sparsity of data points (in number of simulation steps `dx` between data points).
- A `noise/` folder, that contains cases with different amplitudes of added noise to the data points.

Additionally:
- `adjointSW/cases` contains folder `udata`, which corresponds to a case where the inverse problem is solved using velocity measurements instead of height measurements.
- `pinnSW/cases` contains folder `2D`, which corresponds to a 2D case.
---

## Running the cases

To run each case, follow these steps:

1. **Install dependencies**

   This project depends on the [`spooky`](https://github.com/PatricioClark/spooky.git) library, and other python packages.

   run:
    ```bash
    pip install -r requirements.txt

2. **Generate 1D data**

   Go to directory `adjointSW/data/time_marching_swhd1D_DG-scaled-hbnoise` (or `pinnSW/data/time_marching_swhd1D_DG-scaled-hbnoise_pinn` if running PINN case).

   Create directory named `outs`.

   Run `time_marching.py`.

3. **Generate 2D data**

   Go to directory `pinnSW/data/2D`

   Create directory named `outs`.

   Refer to [GHOST](https://github.com/pmininni/GHOST) for details on how to download and compile GHOST (Geophysical High Order Suite for Turbulence by Pablo Mininni). The necesary
   Fortran files are already included in  `src`. The text file `parameters.txt` has the necesary parameters for the current 2D case. Once the code is correctly compiled,
   executable `bin/SWHD` must be copied to `pinnSW/data/2D` and executed (for example using `mpirun -np 1 ./SWHD`).


4. **Run a case**

  a. **Run adjoint case**

  Go to a particular case directory in `adjointSW/cases`(for example `adjointSW/cases/no_noise/dx1`).

  Run `adjoint_GD.py`.

  b. **Run PINN case**
  Go to a particular case directory in `pinnSW/cases`(for example `pinnSW/cases/no_noise/nx1`).

  Create folder named `data`.

  Run `run_pinn.py`.

5. **Plots**

  After running all cases, for both PINNs and Adjoint Method, go to directory `plots`.

  Create folder named `figures`.

  Run each python archive in `plots` directory to save plots of the results in `.pdf` format into `figures` directory.
