export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
export PYTHONPATH="$PYTHONPATH:$PWD/src"
echo "PYTHONPATH=$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

PT="<OUTPUT_DIR>/$FEDERATED_MODEL_TIMESTAMP/round_$FEDERATION_ROUND/global_model_weights.pt"
TEST_OUTPUT_DIR="<OUTPUT_PATH>/inferences/federated/$FEDERATED_MODEL_TIMESTAMP/round_$FEDERATION_ROUND"
TEST_OUTPUT_LOG="$TEST_OUTPUT_DIR/test.log"

DATASET=kitti
CONFIG_DIR="<CONFIGPATH>/v3/kitti_raw.txt" # directory where configs are according to https://github.com/JiawangBian/sc_depth_pl instructions
DATASET_DIR="<DATAPATH>/kitti" # directory where kitti data is according to https://github.com/JiawangBian/sc_depth_pl instructions
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

