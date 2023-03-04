export PYTHONPATH="$PYTHONPATH:$PWD/src"
CONFIG_DIR="/home/eltons-pc/Configurations/v3/kitti_raw.txt"
DATASET_DIR="/home/eltons-pc/Datasets/kitti"
python src/main.py  --config $CONFIG_DIR --dataset_dir $DATASET_DIR --fed_train_num_rounds=3 --fed_train_num_participants=5 --fed_train_num_local_epochs=1 --num_workers=5 --gpu=0 --fed_train_participant_order="sequential" --fed_train_by_drive=1 --fed_train_num_local_train_batches=-1 --fed_train_num_local_val_batches=-1
