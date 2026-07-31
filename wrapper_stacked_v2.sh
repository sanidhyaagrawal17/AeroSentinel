#!/bin/bash
source ~/.bashrc
conda activate cloudspace
cd /teamspace/studios/this_studio/
nohup python train_stacked_v2.py > stacked_v2_train.log 2>&1 &
echo $! > stacked_v2_train.pid
