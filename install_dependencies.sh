#!/bin/bash
sudo apt-get install python3-tk
python3 -m venv ~/py3_venvs/ds
source ~/py3_venvs/ds/bin/activate
pip install pep8
pip install --upgrade autopep8
pip install opencv-contrib-python
pip install numpy
pip install matplotlib
