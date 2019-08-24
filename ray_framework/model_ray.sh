#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --account=rpp-ycoady
#SBATCH --mail-user=benjaminsmith@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=0

module load python/3.6
pip install --user -r requirements.txt

#python3 ray_client.py --mode train --numWorker 4 --data_dir data --save_dir model_output --log_dir log --batch_size 10 --num_epoch 25 --val_intv 1000 --rep_intv 1000 --learn_rate 1e-4
python3 ray_client.py --mode predict --numWorker 4 --data_dir data --save_dir model_output --log_dir log 
