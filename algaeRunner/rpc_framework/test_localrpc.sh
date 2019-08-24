#!/bin/bash

python3 rpyc_worker.py --port 12345 &
python3 rpyc_worker.py --port 12346 &
python3 rpyc_worker.py --port 12347 &

python3 rpyc_client.py --func greyscale --scale 2 --src ../nc --out ../greyscale_output --numproc 6