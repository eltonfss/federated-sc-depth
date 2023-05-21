export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
export PYTHONPATH="$PYTHONPATH:$PWD/src"
echo "PYTHONPATH=$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# 10_participants_5_per_round_12_rounds_1_local_epochs_iid
#FEDERATED_MODEL_TIMESTAMP=09_05_2023_10:30:05
#FEDERATION_ROUND=7

# 9_participants_3_per_round_12_rounds_1_local_epochs_iid
#FEDERATED_MODEL_TIMESTAMP=10_05_2023_21:58:21
#FEDERATION_ROUND=11

# 10_participants_10_per_round_12_rounds_1_local_epochs_iid
#FEDERATED_MODEL_TIMESTAMP=13_05_2023_22:52:57
#FEDERATION_ROUND=9

# 5_participants_5_per_round_12_rounds_1_local_epochs_iid
#FEDERATED_MODEL_TIMESTAMP=21_05_2023_11:54:28
#FEDERATION_ROUND=TBD

# 10_participants_5_per_round_12_rounds_1_local_epochs_by_drive
#FEDERATED_MODEL_TIMESTAMP=08_05_2023_23:18:52
#FEDERATION_ROUND=10

# 9_participants_3_per_round_12_rounds_1_local_epochs_by_drive
#FEDERATED_MODEL_TIMESTAMP=09_05_2023_22:03:14
#FEDERATION_ROUND=9

# 10_participants_10_per_round_12_rounds_1_local_epochs_by_drive
#FEDERATED_MODEL_TIMESTAMP=12_05_2023_19:50:36
#FEDERATION_ROUND=11

# 5_participants_5_per_round_12_rounds_1_local_epochs_by_drive
FEDERATED_MODEL_TIMESTAMP=11_05_2023_22:01:55
FEDERATION_ROUND=11

#CKPT=/home/eltons-pc/Logs/centralized_sc_depth/02_05_2023/last.ckpt
PT="/home/eltons-pc/Backup/Logs/federated-sc-depth/$FEDERATED_MODEL_TIMESTAMP/round_$FEDERATION_ROUND/global_model_weights.pt"
#TEST_OUTPUT_DIR="/home/eltons-pc/Logs/inferences/centralized"
TEST_OUTPUT_DIR="/home/eltons-pc/Logs/inferences/federated/$FEDERATED_MODEL_TIMESTAMP/round_$FEDERATION_ROUND"
TEST_OUTPUT_LOG="$TEST_OUTPUT_DIR/test.log"

DATASET=kitti
CONFIG_DIR="/home/eltons-pc/Configurations/v3/kitti_raw.txt"
DATASET_DIR="/home/eltons-pc/Datasets/$DATASET"
TEST_INPUT_DIR=$DATASET_DIR/testing/color
TEST_GT_DIR=$DATASET_DIR/testing/depth
TEST_SEG_MASK=$DATASET_DIR/testing/seg_mask
DEPTH_PREDICTIONS_DIR="$TEST_OUTPUT_DIR/model_v3/depth"

python src/tests/test.py \
--config $CONFIG_DIR --dataset_dir $DATASET_DIR \
--ckpt_path "$CKPT" --pt_path "$PT"

python src/tests/inference.py --config $CONFIG_DIR \
--input_dir $TEST_INPUT_DIR --output_dir $TEST_OUTPUT_DIR \
--ckpt_path "$CKPT" --pt_path "$PT" \
--save-vis --save-depth

python src/tests/eval_depth.py \
--dataset $DATASET \
--pred_depth=$DEPTH_PREDICTIONS_DIR \
--gt_depth=$TEST_GT_DIR \
--seg_mask=$TEST_SEG_MASK > $TEST_OUTPUT_LOG

