#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --account=rpp-ycoady
#SBATCH --ntasks=17
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=3G
#SBATCH --mail-user=benjaminsmith@uvic.ca
#SBATCH --mail-type=ALL

module load python/3.6
pip install --user netcdf4
pip install --user pillow
pip install --user mpi4py

#greyscale
python3 framework/local_client.py --numWorker 16 --func greyscale --scale 2 --src OLCI/mosaic_output6 --out OLCI/grey6 --numproc 6
