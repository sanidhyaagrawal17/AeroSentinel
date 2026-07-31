#!/bin/zsh
source ~/.zshrc
cd /teamspace/studios/this_studio
python patch_bifpn_remote.py
python train_bifpn.py
