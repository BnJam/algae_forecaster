#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=rpp-ycoady
#SBATCH --mem-per-cpu=3G
#SBATCH --mail-user=benjaminsmith@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --tasks=13
#SBATCH --cpus-per-task=6

module load python/3.6
pip install --user ray
pip install --user netcdf4
pip install --user pillow
pip install --user psutil


python3 framework/ray_client_2.py --numWorker 12 --func greyscale --scale 2 --src OLCI/mosaic_output9 --out OLCI/grey9 --numproc

# Pass the total number of allocated CPUs