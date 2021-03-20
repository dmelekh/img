#!/bin/bash
sudo apt-get install python3-tk
python3 -m venv ~/py3_venvs/ds
cd ~/py3_venvs/ds/bin
source ./activate
pip install opencv-contrib-python
pip install numpy
pip install matplotlib

