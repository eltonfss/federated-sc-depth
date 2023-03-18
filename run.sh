export PYTHONPATH="$PYTHONPATH:$PWD/src"
CONFIG_DIR="/home/eltons-pc/Configurations/v3/kitti_raw.txt"
DATASET_DIR="/home/eltons-pc/Datasets/kitti"
OUTPUT_DIR="/home/eltons-pc/Logs/federated-sc-depth"
MAX_LOCAL_TRAIN_BATCHES=100
MAX_LOCAL_VAL_BATCHES=-1
PARTICIPANT_SORTING="sequential"
DISTRIBUTE_DATASET_BY_DRIVE=1
NUM_ROUNDS=3
NUM_PARTICIPANTS=2
FED_TRAIN_NUM_EPOCHS=1
NUM_WORKERS=5
python src/main.py --config $CONFIG_DIR --dataset_dir $DATASET_DIR --fed_train_num_rounds=$NUM_ROUNDS \
--fed_train_num_participants=$NUM_PARTICIPANTS --fed_train_num_local_epochs=$FED_TRAIN_NUM_EPOCHS \
--num_workers=$NUM_WORKERS --fed_train_participant_order=$PARTICIPANT_SORTING \
--fed_train_by_drive=$DISTRIBUTE_DATASET_BY_DRIVE --fed_train_num_local_train_batches=$MAX_LOCAL_TRAIN_BATCHES \
--fed_train_num_local_val_batches=$MAX_LOCAL_VAL_BATCHES --fed_train_state_save_dir=$OUTPUT_DIR --gpu=0
