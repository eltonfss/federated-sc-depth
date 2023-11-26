export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
export PYTHONPATH="$PYTHONPATH:$PWD/src"
echo "PYTHONPATH=$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# (BOFEDSCDEPTH-KITTI) DISTRIBUTION = IID; TOTAL PARTICIPANTS = 12 ; PARTICIPANTS PER ROUND = 4 ; LOCAL EPOCHS = 3; SEARCH RANGE = 6; MAX LOCAL BATCHES TRAIN = 1000; MAX LOCAL BATCHES VAL = -1; MAX_ROUNDS = 36 ;
#FEDERATED_MODEL_TIMESTAMP='03_09_2023_13:07:11'
#FEDERATION_ROUND=8

# (BOFEDSCDEPTH-DDAD) DISTRIBUTION = NIID; TOTAL PARTICIPANTS = 12 ; PARTICIPANTS PER ROUND = 4 ; LOCAL EPOCHS = 3; SEARCH RANGE = 6; MAX LOCAL BATCHES TRAIN = 1000; MAX LOCAL BATCHES VAL = -1; MAX_ROUNDS = 36 ;
#FEDERATED_MODEL_TIMESTAMP='09_10_2023_22:04:30'
#FEDERATION_ROUND=11

# FEDERATED
#PT="/home/eltons-pc/Logs/federated-sc-depth/$FEDERATED_MODEL_TIMESTAMP/round_$FEDERATION_ROUND/global_model_weights.pt"
#TEST_OUTPUT_DIR="/home/eltons-pc/Logs/inferences/federated/$FEDERATED_MODEL_TIMESTAMP/round_$FEDERATION_ROUND"

# CENTRALIZED KITTI
CKPT=/home/eltons-pc/Logs/centralized_sc_depth/kitti/02_05_2023/last.ckpt
TEST_OUTPUT_DIR="/home/eltons-pc/Logs/inferences/centralized/02_05_2023"

# CENTRALIZED DDAD
#CKPT=/home/eltons-pc/Logs/centralized_sc_depth/ddad/10_06_2023/last.ckpt
#TEST_OUTPUT_DIR="/home/eltons-pc/Logs/inferences/centralized/10_06_2023"

TEST_OUTPUT_LOG="$TEST_OUTPUT_DIR/test.log"

# DATASET KITTI
DATASET=kitti
CONFIG_DIR="/home/eltons-pc/Configurations/v3/kitti_raw.txt" # KITTI

# DATASET DDAD
#DATASET=ddad
#CONFIG_DIR="/home/eltons-pc/Configurations/v3/ddad.txt" # DDAD

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

