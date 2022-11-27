# absolute path that contains all datasets
DATA_ROOT=/home/jussarafigueiredodesouzasoares/sc_depth_pl/data

# # kitti
DATASET=$DATA_ROOT/kitti
CONFIG=/home/jussarafigueiredodesouzasoares/sc_depth_pl/configs/v3/kitti_raw.txt

# # nyu
# DATASET=$DATA_ROOT/nyu
# CONFIG=configs/v2/nyu.txt

# ddad
#DATASET=$DATA_ROOT/ddad
#CONFIG=configs/v3/ddad.txt

# run
export CUDA_VISIBLE_DEVICES=0
python train.py --config $CONFIG --dataset_dir $DATASET --epoch_size=1 --val_epoch_size=1 --num_epochs=1