#!/bin/bash
pkill -f 'train_true_bifpn.py' || true
source /home/zeus/miniconda3/etc/profile.d/conda.sh
conda activate cloudspace
nohup python train_true_bifpn.py > true_bifpn_train2.log 2>&1 &
