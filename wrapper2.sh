#!/bin/zsh
source ~/.zshrc
cd /teamspace/studios/this_studio
pip install -q pycocotools
python run_ensemble_experiment.py
cat ensemble_report.md
