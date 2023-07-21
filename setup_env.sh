#!/bin/sh

apt upgrade
apt-get update

conda config --set default_threads 4
conda config --set safety_checks enabled
conda config --set channel_priority strict

curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"

bash ./Mambaforge-$(uname)-$(uname -m).sh -b -f

source ~/.bashrc

mamba update conda --yes
mamba update conda --yes


mamba create -n gad python=3.9 -y

mamba activate gad
mamba install -c conda ipykernel --yes
pip install -r ./requirements.txt
python -m ipykernel install --user --name=gad

rm "./Mambaforge-$(uname)-$(uname -m).sh"