#!/usr/bin/env bash

set -x

CONFIG=$1

/home/chenjunfen/workspace/XZH/env/miniconda3/envs/arskl/bin/python $(dirname "$0")/train.py $CONFIG
