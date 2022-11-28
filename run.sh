export PYTHONPATH="$PYTHONPATH:$PWD/src"
CONFIG_DIR="$PWD/configs/v3/kitti_raw.txt"
DATASET_DIR="$PWD/kitti_dataset"
python src/main.py  --config $CONFIG_DIR --dataset_dir $DATASET_DIR --fed_train_num_rounds=1 --fed_train_num_participants=4 --fed_train_num_local_epochs=1 --fed_train_num_local_train_batches=1 --fed_train_num_local_val_batches=1 --num_workers=3