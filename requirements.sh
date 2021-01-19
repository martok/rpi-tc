#!/usr/bin/env bash

set -eu

sudo apt install python3-opencv
sudo apt install libatlas3-base libgfortran5         # for piwheels-numpy

python3 -m venv --clear --system-site-packages venv
. venv/bin/activate
pip install -U -r requirements.txt

