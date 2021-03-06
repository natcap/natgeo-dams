#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/usr/local/tf_models/
MODEL_DIR=/usr/local/model_files
python /usr/local/tf_models/official/vision/detection/main.py \
  --strategy_type=one_device \
  --num_cpus=1 \
  --model_dir="${MODEL_DIR?}" \
  --mode=train \
  --config_file="dams_retinanet.yaml"
