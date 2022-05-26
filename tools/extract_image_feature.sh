#!/usr/bin/env bash

set -x

DATA_DIR=$1
WORK_DIR=$2
CAMERA_ID=$3
SPLIT=$4

python3 -u $(dirname $0)/extract_image_feature.py $DATA_DIR $WORK_DIR $CAMERA_ID $SPLIT