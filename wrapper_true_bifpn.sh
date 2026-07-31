#!/bin/bash
source /home/zeus/miniconda3/etc/profile.d/conda.sh
conda activate cloudspace

echo "Reinstalling clean ultralytics to revert WIoU patch..."
pip install --force-reinstall ultralytics==8.4.110

echo "Running BiFPN architecture patch (skipping metrics/loss)..."
python patch_bifpn_remote.py

echo "Starting True BiFPN training..."
python train_true_bifpn.py
