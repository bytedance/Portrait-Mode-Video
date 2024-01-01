#!/bin/bash
set -x

pip3 uninstall slowfast
pip3 uninstall slowfast

cd Uniformer
rm -rf build/
python3 setup.py build develop
cd -

chmod +x ${EXP_EXCUATION_SCRIPT}

./${EXP_EXCUATION_SCRIPT} \
  DATA.PM_SUBSET "''" \
  DATA.LABEL_PATH_TEMPLATE "{}{}.csv" "${@}" 

./${EXP_EXCUATION_SCRIPT} \
  TRAIN.ENABLE False \
  DATA.PM_SUBSET "''" \
  DATA.LABEL_PATH_TEMPLATE "{}{}.csv" \
  DATA.TRAIN_JITTER_SCALES [224,224] \
  DATA.TEST_CROP_SIZE 224 \
  TEST.NUM_ENSEMBLE_VIEWS 4 \
  DATA_LOADER.NUM_WORKERS 5 \
  TEST.PROCESS True \
  TEST.NUM_SPATIAL_CROPS 1 "${@}"
