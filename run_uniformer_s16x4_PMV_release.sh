#!/bin/bash

cd Uniformer/
PYTHONPATH=$PYTHONPATH:slowfast \
output_dir=${UPLOAD_SOURCE}

work_path=exp/uniformer_s16x4_k400
ckpt_path=$output_dir

mkdir -p $ckpt_path

CUR_DIR=$(cd $(dirname $0); pwd)
cd $CUR_DIR

python3 tools/run_net.py \
  --cfg $work_path/config.yaml \
  --init_method tcp://$WORKER_0_HOST:28999 \
  --num_shards ${WORKER_NUM} \
  --shard_id ${WORKER_ID} \
  SOLVER.BASE_LR_SCALE_NUM_SHARDS True \
  DATA.PATH_TO_DATA_DIR ./data_list/PMV \
  DATA.PATH_LABEL_SEPARATOR "," \
  MODEL.NUM_CLASSES 400 \
  DATA_LOADER.NUM_WORKERS 5 \
  TRAIN.EVAL_PERIOD 5 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  TRAIN.BATCH_SIZE 96 \
  NUM_GPUS 8 \
  UNIFORMER.DROP_DEPTH_RATE 0.1 \
  SOLVER.MAX_EPOCH 110 \
  SOLVER.BASE_LR 3e-4 \
  SOLVER.WARMUP_EPOCHS 10.0 \
  DATA.TEST_CROP_SIZE 224 \
  TEST.NUM_ENSEMBLE_VIEWS 1 \
  TEST.NUM_SPATIAL_CROPS 1 \
  RNG_SEED 6666 \
  MODEL.USE_CHECKPOINT True \
  OUTPUT_DIR $ckpt_path \
  TENSORBOARD.ENABLE True \
  MODEL.CHECKPOINT_NUM [0,0,1,0] $@