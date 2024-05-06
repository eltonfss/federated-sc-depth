export PYTORCH_CUDA_ALLOC_CONF=cudaMallocAsync,garbage_collection_threshold:0.6,max_split_size_mb:128
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
export PYTHONPATH="$PYTHONPATH:$PWD/src"
echo "PYTHONPATH=$PYTHONPATH"

MODEL_VERSION="v3"

# KITTI
CONFIG_DIR="/home/eltons-pc/Configurations/v3/kitti_raw.txt"
DATASET_DIR="/home/eltons-pc/Datasets/kitti"
#SOURCE_MODEL_PATH="/home/eltons-pc/Logs/federated-sc-depth/15_11_2023_02:38:56/round_3/global_model_weights.pt"

# DDAD
#CONFIG_DIR="/home/eltons-pc/Configurations/v3/ddad.txt"
#DATASET_DIR="/home/eltons-pc/Datasets/ddad"
SOURCE_MODEL_PATH="/home/eltons-pc/Logs/federated-sc-depth/07_12_2023_00:00:26/round_3/global_model_weights.pt"

OUTPUT_DIR="/home/eltons-pc/Logs/federated-sc-depth"
RESTORE_DIR="$OUTPUT_DIR/05_05_2024_22:21:52"
MAX_LOCAL_TRAIN_BATCHES=1000
MAX_LOCAL_VAL_BATCHES=-1
PARTICIPANT_SORTING="sequential" #IID
#PARTICIPANT_SORTING="random" # BY DRIVE
DISTRIBUTE_DATASET_BY_DRIVE=0 # IID
#DISTRIBUTE_DATASET_BY_DRIVE=1 # BY DRIVE
DISTRIBUTE_DATASET_BY_DRIVE_SORT="eager"
DISTRIBUTE_DATASET_BY_DRIVE_REDISTRIBUTE_REMAINING=1
NUM_ROUNDS=36
NUM_PARTICIPANTS=12
NUM_WORKERS=8

# LOCAL EPOCHS
#FED_TRAIN_NUM_EPOCHS=1
#FED_TRAIN_NUM_EPOCHS=2
FED_TRAIN_NUM_EPOCHS=3

# PARTICIPATION RATION
FRAC_PARTICIPANTS_PER_ROUND=0.3333333333333333 # 1/3
#FRAC_PARTICIPANTS_PER_ROUND=0.5 # 1/2
#FRAC_PARTICIPANTS_PER_ROUND=1 # 1/1

#FED_TRAIN_SKIP_BAD_ROUNDS=0 #FedSCDepth
FED_TRAIN_SKIP_BAD_ROUNDS=1 #BOFedSCDepth

#FED_TRAIN_AVG_SEARCH_RANGE=-1 #FedSCDepth
FED_TRAIN_AVG_SEARCH_RANGE=6 #BOFedSCDepth

#FED_TRAIN_AVG_SEARCH_STRATEGY="" #FedSCDepth
FED_TRAIN_AVG_SEARCH_STRATEGY="BayesianOptimization" #BOFedSCDepth

# EXPERIENCE REPLAY BUFFER
#MODEL_VERSION="v3_with_er"
REPLAY_DATASET_DIR="/home/eltons-pc/Datasets/ddad"
REPLAY_DATASET_NAME=ddad
ER_BUFFER_SIZE=200
ER_SIZE=1
ER_FREQUENCY=1
ER_MODE='combined_loss_with_retrain' # none, combined_loss, retrain, combined_loss_with_retrain

python src/main.py --config $CONFIG_DIR --dataset_dir $DATASET_DIR --fed_train_num_rounds=$NUM_ROUNDS \
--fed_train_num_participants=$NUM_PARTICIPANTS --fed_train_num_local_epochs=$FED_TRAIN_NUM_EPOCHS \
--num_workers=$NUM_WORKERS --fed_train_participant_order=$PARTICIPANT_SORTING \
--fed_train_by_drive_redistribute_remaining=$DISTRIBUTE_DATASET_BY_DRIVE_REDISTRIBUTE_REMAINING \
--fed_train_by_drive=$DISTRIBUTE_DATASET_BY_DRIVE --fed_train_num_local_train_batches=$MAX_LOCAL_TRAIN_BATCHES \
--fed_train_by_drive_sort=$DISTRIBUTE_DATASET_BY_DRIVE_SORT \
--fed_train_num_local_val_batches=$MAX_LOCAL_VAL_BATCHES --fed_train_state_backup_dir=$OUTPUT_DIR --gpu=0 \
--fed_train_frac_participants_per_round=$FRAC_PARTICIPANTS_PER_ROUND \
--fed_train_skip_bad_rounds=$FED_TRAIN_SKIP_BAD_ROUNDS \
--fed_train_average_search_strategy=$FED_TRAIN_AVG_SEARCH_STRATEGY \
--fed_train_average_search_range=$FED_TRAIN_AVG_SEARCH_RANGE \
--fed_train_state_restore_dir=$RESTORE_DIR \
--pt_path=$SOURCE_MODEL_PATH \
--model_version=$MODEL_VERSION \
--replay_dataset_dir=$REPLAY_DATASET_DIR \
--replay_dataset_name=$REPLAY_DATASET_NAME \
--er_buffer_size=$ER_BUFFER_SIZE \
--er_size=$ER_SIZE \
--er_frequency=$ER_FREQUENCY \
--global_replay_mode=$ER_MODE

