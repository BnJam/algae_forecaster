#!/bin/bash

python3 local_client.py --mode grey --gnumWorker 4 --gsrc ../nc --gout ../greyscale_output --gnumproc 6
#python3 local_client.py --mode scale --snumWorker 4 --sscale 4 --ssrc ../greyscale_output --out ../scaleImgOutput --snumproc 6