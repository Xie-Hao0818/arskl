#!/usr/bin/env bash
set -x

CONFIG=$1

MKL_SERVICE_FORCE_INTEL=1
/root/miniconda3/envs/arskl/bin/python $(dirname "$0")/train.py $CONFIG