Code to generate data points in `data/time_marching_swhd1D_DG-scaled-hbnoise`, which are used in all cases in `cases/`.

The directory `cases/` contains:
- A `no_noise/` folder, which includes several subdirectories named `dx*`, where each number indicates how sparse the data points are (in number of simulation steps `dx` between data points).
- A `noise/` folder, which contains cases with different amplitudes of added noise to the data points.

---

## Running the cases

To run each case, follow these steps:

1. **Install `pySPEC`**

   This project depends on the [`pySPEC`](https://github.com/PatricioClark/pySPEC) library, and other python packages.

   run:
    ```bash
    pip install -r requirements.txt





2. **Generate data**
   Stand in directory `data/time_marching_swhd1D_DG-scaled-hbnoise` by running:

   ```bash
   cd ./data/time_marching_swhd1D_DG-scaled-hbnoise.

   Run `time_marching.py`.

3. **Run a case**
   stand in directory `cases/` + path to particular case.

   Once there, run `adjoint_GD.py`.
