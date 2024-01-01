export PYTHONPATH=$PYTHONPATH:./slowfast

sleep 1d

python3 tools/run_net.py \
--cfg /opt/tiger/projects/SlowFast/configs/Kinetics/I3D_8x8_R50_IN1K.yaml \
--opts DATA.PATH_TO_DATA_DIR /opt/tiger/projects/SlowFast/data_list/3massiv \
DATA.PATH_PREFIX /mnt/bn/kinetics-pl-hl/kinetics-pl-hl/3massiv/ \
DATA.PATH_LABEL_SEPARATOR "," \
OUTPUT_DIR /mnt/bn/kinetics-pl-hl/kinetics-pl-hl/output/temp \
TRAIN.CHECKPOINT_FILE_PATH /mnt/bn/kinetics-pl-hl/kinetics-pl-hl/pretrained_model/I3D_8x8_R50.pkl \
MODEL.NUM_CLASSES 34 \
  TRAIN.CHECKPOINT_TYPE caffe2 \
DATA.DECODING_BACKEND 'pyav'

