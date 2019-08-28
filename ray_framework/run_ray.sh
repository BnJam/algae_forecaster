#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=rpp-ycoady
#SBATCH --mem-per-cpu=3G
#SBATCH --mail-user=benjaminsmith@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --tasks=13
#SBATCH --cpus-per-task=6

module load python/3.6
pip install --user -r requirements.txt


# run grey scale with ray
python3 framework/ray_client_2.py --mode grey --numWorker 12 --gsrc OLCI/mosaic_output --gout OLCI/grey

# run image scaling with rauy
#python3 framework/ray_client_2.py --mode scale --scale 30 --numWorker 12 --ssrc OLCI/mosaic_output --sout OLCI/grey
