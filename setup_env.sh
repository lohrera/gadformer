#!/bin/sh

apt-get update

source ~/.bashrc

conda update conda -y

conda create -n gad -y
conda update conda --yes
conda update conda --yes

conda init bash
source ~/.bashrc
conda activate gad
conda install -c conda ipykernel --yes
pip install -r ./requirements.txt

python -m ipykernel install --user --name=gad
