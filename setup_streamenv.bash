#!/usr/bin/bash

python3 -m venv --system-site-packages ./venv
. ./venv/bin/activate
./venv/bin/python3 -m pip install --upgrade pip
pip install -r ./requirements.txt
