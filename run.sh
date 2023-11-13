export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
export PYTHONPATH="$PYTHONPATH:$PWD/src"
echo "PYTHONPATH=$PYTHONPATH"
CONFIG_DIR="<CONFIGPATH>/v3/kitti_raw.txt" # directory where configs are according to https://github.com/JiawangBian/sc_depth_pl instructions
DATASET_DIR="<DATAPATH>/kitti" # directory where kitti data is according to https://github.com/JiawangBian/sc_depth_pl instructions
OUTPUT_DIR="" # directory where output logs and models should be saved (including checkpoints)
RESTORE_DIR="" # directory where preexisting federated training should be restarted from
MAX_LOCAL_TRAIN_BATCHES=1000
MAX_LOCAL_VAL_BATCHES=-1
#PARTICIPANT_SORTING="sequential" # For IID Training
PARTICIPANT_SORTING="random" # For Non-IDD (BY DRIVE) Training
#DISTRIBUTE_DATASET_BY_DRIVE=0 # For IID Training
DISTRIBUTE_DATASET_BY_DRIVE=1 # For Non-IID (BY DRIVE) Training
DISTRIBUTE_DATASET_BY_DRIVE_SORT="eager" # For Non-IID (BY DRIVE) Training
DISTRIBUTE_DATASET_BY_DRIVE_REDISTRIBUTE_REMAINING=1 # For Non-IID (BY DRIVE) Training
NUM_ROUNDS=30 
NUM_PARTICIPANTS=10
FRAC_PARTICIPANTS_PER_ROUND=0.5
FED_TRAIN_NUM_EPOCHS=3
NUM_WORKERS=8
python src/main.py --config $CONFIG_DIR --dataset_dir $DATASET_DIR --fed_train_num_rounds=$NUM_ROUNDS \
--fed_train_num_participants=$NUM_PARTICIPANTS --fed_train_num_local_epochs=$FED_TRAIN_NUM_EPOCHS \
--num_workers=$NUM_WORKERS --fed_train_participant_order=$PARTICIPANT_SORTING \
--fed_train_by_drive_redistribute_remaining=$DISTRIBUTE_DATASET_BY_DRIVE_REDISTRIBUTE_REMAINING \
--fed_train_by_drive=$DISTRIBUTE_DATASET_BY_DRIVE --fed_train_num_local_train_batches=$MAX_LOCAL_TRAIN_BATCHES \
--fed_train_by_drive_sort=$DISTRIBUTE_DATASET_BY_DRIVE_SORT \
--fed_train_num_local_val_batches=$MAX_LOCAL_VAL_BATCHES --fed_train_state_backup_dir=$OUTPUT_DIR --gpu=0 \
--fed_train_frac_participants_per_round=$FRAC_PARTICIPANTS_PER_ROUND \
--fed_train_state_restore_dir=$RESTORE_DIR

