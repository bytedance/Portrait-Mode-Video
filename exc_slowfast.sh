#!/bin/bash
set -x

pip3 uninstall slowfast
pip3 uninstall slowfast

cd MViT
rm -rf build/
python3 setup.py build develop
pip3 uninstall -y pytorchvideo
pip3 uninstall -y pytorchvideo
pip3 uninstall -y pytorchvideo
cd -

cd pytorchvideo
pip3 install -e .
cd -

pip3 install -e detectron2_repo
pip3 install librosa


echo "In projects/run.sh Command: ${@}"
chmod +x ${EXP_EXCUATION_SCRIPT}
./${EXP_EXCUATION_SCRIPT} \
  DATA.PM_SUBSET "''" \
  TENSORBOARD.ENABLE True \
  DATA.LABEL_PATH_TEMPLATE "{}{}.csv" "${@}"
