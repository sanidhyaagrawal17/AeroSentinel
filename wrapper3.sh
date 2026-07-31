#!/bin/zsh
source ~/.zshrc
cd /teamspace/studios/this_studio
python patch_siou.py
python train_siou.py
