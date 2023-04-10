export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
export PYTHONPATH="$PYTHONPATH:$PWD/src"
echo "PYTHONPATH=$PYTHONPATH"
CONFIG_DIR="/home/eltons-pc/Configurations/v3/kitti_raw.txt"
DATASET_DIR="/home/eltons-pc/Datasets/kitti"
OUTPUT_DIR="/home/eltons-pc/Logs/federated-sc-depth"
#RESTORE_DIR="$OUTPUT_DIR/10_04_2023_00:49:33"
MAX_LOCAL_TRAIN_BATCHES=100
MAX_LOCAL_VAL_BATCHES=100
PARTICIPANT_SORTING="sequential"
DISTRIBUTE_DATASET_BY_DRIVE=0
NUM_ROUNDS=50
NUM_PARTICIPANTS=34
FRAC_PARTICIPANTS_PER_ROUND=0.15
FED_TRAIN_NUM_EPOCHS=3
NUM_WORKERS=8
python src/main.py --config $CONFIG_DIR --dataset_dir $DATASET_DIR --fed_train_num_rounds=$NUM_ROUNDS \
--fed_train_num_participants=$NUM_PARTICIPANTS --fed_train_num_local_epochs=$FED_TRAIN_NUM_EPOCHS \
--num_workers=$NUM_WORKERS --fed_train_participant_order=$PARTICIPANT_SORTING \
--fed_train_by_drive=$DISTRIBUTE_DATASET_BY_DRIVE --fed_train_num_local_train_batches=$MAX_LOCAL_TRAIN_BATCHES \
--fed_train_num_local_val_batches=$MAX_LOCAL_VAL_BATCHES --fed_train_state_backup_dir=$OUTPUT_DIR --gpu=0 \
--fed_train_frac_participants_per_round=$FRAC_PARTICIPANTS_PER_ROUND \
#--fed_train_state_restore_dir=$RESTORE_DIR
