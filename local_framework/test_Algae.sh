#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --account=rpp-ycoady
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --mail-user=benjaminsmith@uvic.ca
#SBATCH --mail-type=ALL


#greyscale
python3 local_framework/local_client.py --numWorker 12 --func greyscale --scale 2 --src OLCI/mosaic_output --out OLCI/grey --numproc 6
