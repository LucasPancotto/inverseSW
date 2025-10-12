Directory `adjointSW` contains code for Adjoint State method cases, and directory `pinnSW` contains code for PINN cases. Both directories work independently.

For each method, the code to generate data points is contained in `adjointSW/data/time_marching_swhd1D_DG-scaled-hbnoise` and `pinnSW/data/time_marching_swhd1D_DG-scaled-hbnoise_pinn` respectively. The cases for each method are contained in  `adjointSW/cases` and `pinnSW/cases`, respectively.

Each directory `cases/` contains:
- A `no_noise/` folder, that contains several subdirectories which are cases with different sparsity of data points (in number of simulation steps `dx` between data points).
- A `noise/` folder, that contains cases with different amplitudes of added noise to the data points.

---

## Running the cases

To run each case, follow these steps:

1. **Install dependencies**

   This project depends on the [`pySPEC`](https://github.com/PatricioClark/pySPEC) library, and other python packages.

   run:
    ```bash
    pip install -r requirements.txt

2. **Generate data**

   Go to directory `data/time_marching_swhd1D_DG-scaled-hbnoise` (or `data/time_marching_swhd1D_DG-scaled-hbnoise_pinn` if running PINN case).

   Create directory named `outs`.

   Run `time_marching.py`.

3. **Run a case**

  a. **Run adjoint case**

  Go to a particular case directory in `adjointSW/cases`(for example `adjointSW/cases/no_noise/dx1`).

  Run `adjoint_GD.py`.

  b. **Run PINN case**
  Go to a particular case directory in `pinnSW/cases`(for example `pinnSW/cases/no_noise/nx1`).

  Create folder named `data`.

  Run `run_pinn.py`.
