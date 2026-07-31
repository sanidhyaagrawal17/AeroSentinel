#!/bin/bash
source ~/.bashrc
conda activate cloudspace
cd /teamspace/studios/this_studio/
nohup python tune_rfaconv.py > tune_rfaconv.log 2>&1 &
echo $! > tune_rfaconv.pid
