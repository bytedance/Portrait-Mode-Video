export PYTHONPATH=$PYTHONPATH:./slowfast

BYTENAS_VOLUME_NAME="kinetics-pl-hl/kinetics-pl-hl"

python3 tools/run_net.py \
--cfg exp/uniformer_s8x8_k400/config.yaml \
DATA.PATH_TO_DATA_DIR ./data_list/3massiv \
DATA.PATH_PREFIX /mnt/bn/kinetics-pl-hl/kinetics-pl-hl//3massiv \
DATA.PATH_LABEL_SEPARATOR "," \
TRAIN.EVAL_PERIOD 5 \
TRAIN.CHECKPOINT_PERIOD 1 \
TRAIN.BATCH_SIZE 32 \
NUM_GPUS 2 \
UNIFORMER.DROP_DEPTH_RATE 0.1 \
DATA_LOADER.NUM_WORKERS 5 \
SOLVER.MAX_EPOCH 100 \
SOLVER.BASE_LR 4e-4 \
SOLVER.WARMUP_EPOCHS 5.0 \
DATA.TEST_CROP_SIZE 224 \
TEST.NUM_ENSEMBLE_VIEWS 1 \
TEST.NUM_SPATIAL_CROPS 1 \
RNG_SEED 6666 \
MODEL.NUM_CLASSES 34 \
OUTPUT_DIR /mnt/bn/kinetics-pl-hl/kinetics-pl-hl/output/temp/   
