#!/bin/bash

export CUDA_VISIBLE_DEVICES=4
ulimit -n 160000
python main.py -j 0 --batch-size 64 /home/bala/lemniscate_vis/data/ 
