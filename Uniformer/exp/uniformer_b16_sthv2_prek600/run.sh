work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:./slowfast \
python tools/run_net.py \
  --cfg $work_path/config.yaml \
  DATA.PATH_TO_DATA_DIR ./data_list/sthv2 \
  DATA.PATH_PREFIX your_path_to_data \
  DATA.LABEL_PATH_TEMPLATE "somesomev2_rgb_{}_split.txt" \
  DATA.IMAGE_TEMPLATE "img_{:05d}.jpg" \
  DATA.PATH_LABEL_SEPARATOR "," \
  TRAIN.EVAL_PERIOD 5 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  TRAIN.BATCH_SIZE 40 \
  NUM_GPUS 8 \
  UNIFORMER.DROP_DEPTH_RATE 0.5 \
  SOLVER.MAX_EPOCH 50 \
  SOLVER.BASE_LR 2.5e-4 \
  SOLVER.WARMUP_EPOCHS 5.0 \
  DATA.TEST_CROP_SIZE 224 \
  TEST.NUM_ENSEMBLE_VIEWS 1 \
  TEST.NUM_SPATIAL_CROPS 1 \
  RNG_SEED 6666 \
  OUTPUT_DIR $work_path