#!/bin/sh

cd src
python evaluate_simple.py
python evaluate_alex_net.py
python evaluate_vgg.py
python evaluate_google_net.py
