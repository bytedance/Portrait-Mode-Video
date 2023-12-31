PROJ_DIR="YOU_PROJECT_DIR"

export UPLOAD_SOURCE={PROJ_DIR}/uniformer_X3D_M_PMV_top_merge_0804_part_dedup/scratch_crop_224_default_recipe_bs_64x4_lr0.05_300e/
export EXP_EXCUATION_SCRIPT=run_uniformer_X3D_M_PMV_release.sh

chmod +x exc_uniformer_X3D_M.sh

./exc_uniformer_X3D_M.sh \
DATA_LOADER.NUM_WORKERS 10 \
DATA.DECODING_BACKEND decord \
DATA.TRAIN_CROP_SIZE 224 \
SOLVER.MAX_EPOCH 300 \
TRAIN.BATCH_SIZE 64 \
SOLVER.BASE_LR 0.05 \
NUM_GPUS 8 \
SOLVER.BASE_LR_SCALE_NUM_SHARDS True \
DATA.PM_SUBSET _pmv400 \
DATA.PATH_PREFIX {PROJ_DIR}/PMV_dataset/ \
MODEL.NUM_CLASSES 400 \
DATA.TRAIN_JITTER_SCALES_AUTO_ADJUST False


export UPLOAD_SOURCE={PROJ_DIR}/uniformer_X3D_M_PMV_top_merge_0804_part_dedup/scratch_crop_224_random_scale_recipe_bs_64x4_lr0.05_300e/
export EXP_EXCUATION_SCRIPT=run_uniformer_X3D_M_PMV_release.sh

chmod +x exc_uniformer_X3D_M.sh

./exc_uniformer_X3D_M.sh \
DATA_LOADER.NUM_WORKERS 10 \
DATA.DECODING_BACKEND decord \
DATA.TRAIN_CROP_SIZE 224 \
SOLVER.MAX_EPOCH 300 \
TRAIN.BATCH_SIZE 64 \
SOLVER.BASE_LR 0.05 \
NUM_GPUS 8 \
SOLVER.BASE_LR_SCALE_NUM_SHARDS True \
DATA.PM_SUBSET _pmv400 \
DATA.PATH_PREFIX {PROJ_DIR}/PMV_dataset/ \
MODEL.NUM_CLASSES 400 \
DATA.TRAIN_JITTER_SCALES_AUTO_ADJUST False \
DATA.TRAIN_JITTER_ASPECT_RELATIVE [0.75,1.3333] \
DATA.TRAIN_JITTER_SCALES_RELATIVE [0.08,1.0] 


export UPLOAD_SOURCE={PROJ_DIR}/uniformer_X3D_M_PMV_top_merge_0804_part_dedup/scratch_crop_256_192_default_recipe_bs_64x4_lr0.05_300e/
export EXP_EXCUATION_SCRIPT=run_uniformer_X3D_M_PMV_release.sh

chmod +x exc_uniformer_X3D_M.sh

./exc_uniformer_X3D_M.sh \
DATA_LOADER.NUM_WORKERS 10 \
DATA.DECODING_BACKEND decord \
DATA.TRAIN_CROP_SIZE 224 \
SOLVER.MAX_EPOCH 300 \
TRAIN.BATCH_SIZE 64 \
SOLVER.BASE_LR 0.05 \
NUM_GPUS 8 \
SOLVER.BASE_LR_SCALE_NUM_SHARDS True \
DATA.PM_SUBSET _pmv400 \
DATA.PATH_PREFIX {PROJ_DIR}/PMV_dataset/ \
MODEL.NUM_CLASSES 400 \
DATA.TRAIN_JITTER_SCALES_AUTO_ADJUST False \
DATA.TRAIN_JITTER_SCALES_AUTO_ADJUST True \
DATA.TRAIN_CROP_SIZE_RECT [256,192] \
DATA.TRAIN_JITTER_ASPECT_RELATIVE [] \
DATA.TRAIN_JITTER_SCALES_RELATIVE []


export UPLOAD_SOURCE={PROJ_DIR}/uniformer_X3D_M_PMV_top_merge_0804_part_dedup/scratch_crop_288_192_default_recipe_bs_64x4_lr0.05_300e/
export EXP_EXCUATION_SCRIPT=run_uniformer_X3D_M_PMV_release.sh

chmod +x exc_uniformer_X3D_M.sh

./exc_uniformer_X3D_M.sh \
DATA_LOADER.NUM_WORKERS 10 \
DATA.DECODING_BACKEND decord \
DATA.TRAIN_CROP_SIZE 224 \
SOLVER.MAX_EPOCH 300 \
TRAIN.BATCH_SIZE 64 \
SOLVER.BASE_LR 0.05 \
NUM_GPUS 8 \
SOLVER.BASE_LR_SCALE_NUM_SHARDS True \
DATA.PM_SUBSET _pmv400 \
DATA.PATH_PREFIX {PROJ_DIR}/PMV_dataset/ \
MODEL.NUM_CLASSES 400 \
DATA.TRAIN_JITTER_SCALES_AUTO_ADJUST False \
DATA.TRAIN_JITTER_SCALES_AUTO_ADJUST True \
DATA.TRAIN_CROP_SIZE_RECT [288,192] \
DATA.TRAIN_JITTER_ASPECT_RELATIVE [] \
DATA.TRAIN_JITTER_SCALES_RELATIVE []