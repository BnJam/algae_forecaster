#!/bin/bash

#python3 ray_client.py --mode grey --numWorker 2 --gsrc ../nc --gout ../greyscale_output
python3 ray_client.py --mode scale --scale 2 --numWorker 2 --ssrc ../greyscale_output --sout ../data/valid/algae
#python3 ray_client.py --mode train --numWorker 2 --data_dir ../scaleImgOutput --save_dir train_output --log_dir ../log --batch_size 1 --num_epoch 1 --val_intv 100 --rep_intv 100 --learn_rate 1e-2