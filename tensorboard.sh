#!/bin/sh

# Get directory of this file
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "\nStarting TensorBoard with the following log directory: ${DIR}/logs"
tensorboard --logdir="${DIR}/logs"
