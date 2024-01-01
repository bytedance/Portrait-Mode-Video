#!/bin/bash

cd Uniformer/
PYTHONPATH=$PYTHONPATH:slowfast \
output_dir=${UPLOAD_SOURCE}

work_path=configs/Kinetics/
ckpt_path=$output_dir
exp_name=${EXP_NAME}

mkdir -p $ckpt_path

CUR_DIR=$(cd $(dirname $0); pwd)
cd $CUR_DIR

python3 tools/run_net.py \
  --cfg $work_path/X3D_M.yaml \
  --init_method tcp://$WORKER_0_HOST:28999 \
  --num_shards ${WORKER_NUM} \
  --shard_id ${WORKER_ID} \
  SOLVER.BASE_LR_SCALE_NUM_SHARDS True \
  DATA.PATH_TO_DATA_DIR ./data_list/PMV \
  DATA.PATH_LABEL_SEPARATOR "," \
  MODEL.NUM_CLASSES 232 \
  DATA_LOADER.NUM_WORKERS 16 \
  TEST.NUM_SPATIAL_CROPS 1 \
  DATA.TEST_CROP_SIZE 224 \
  NUM_GPUS 8 \
  OUTPUT_DIR $ckpt_path $@