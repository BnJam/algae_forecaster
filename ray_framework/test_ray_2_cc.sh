#!/bin/bash
#SBATCH --account=rpp-ycoady
#SBATCH --mail-user=benjaminsmith@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=3GB
#SBATCH --ntasks=13
#SBATCH --cpus-per-task=6

module load python/3.6
pip install --user psutil
pip install --user ray
pip install --user mpi4py
pip install --user netcdf4
pip install --user pillow
pip install --user numpy

python3 framework/ray_client.py --numWorker 12 --func greyscale --scale 2 --src OLCI/mosaic_output --out OLCI/grey --numproc 6