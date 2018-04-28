#!/bin/sh

cd src
python train_simple.py
python train_alex_net.py
python train_vgg.py
python train_google_net.py
