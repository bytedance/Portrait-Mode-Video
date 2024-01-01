#!/bin/bash
set -x

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
  DATA.TEST_CROP_SIZE 256 \
  TEST.PROCESS True \
  TEST.NUM_SPATIAL_CROPS 3 "${@}"
