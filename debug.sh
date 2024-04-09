export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
export PYTHONPATH="$PYTHONPATH:$PWD/src"
echo "PYTHONPATH=$PYTHONPATH"

python src/main.py --config="/home/eltons-pc/Configurations/v3/kitti_raw.txt" \
--dataset_dir="/home/eltons-pc/Datasets/kitti" \
--dataset_name=kitti \
--model_version="v3_with_er" \
--replay_dataset_dir="/home/eltons-pc/Configurations/v3/ddad.txt" \
--replay_dataset_name=ddad \
--er_buffer_size=5 \
--er_size=5 \
--er_frequency=5 \
--fed_train_num_rounds=50 \
--fed_train_num_participants=3 \
--fed_train_frac_participants_per_round=1 \
--fed_train_num_local_epochs=3 \
--num_workers=8 \
--fed_train_participant_order="random" \
--fed_train_by_drive=1 \
--fed_train_by_drive_sort="eager" \
--fed_train_by_drive_redistribute_remaining=1 \
--gpu=0 \
--fed_train_average_search_range=-1 \
--fed_train_num_local_train_batches=20 \
--fed_train_num_local_val_batches=10 \
--fed_train_average_search_strategy="" \
--fed_train_state_backup_dir="/home/eltons-pc/Logs/federated-sc-depth" \
--pt_path="/home/eltons-pc/Logs/federated-sc-depth/07_12_2023_00:00:26/round_3/global_model_weights.pt"
#\
#--fed_train_state_restore_dir="/home/eltons-pc/Logs/federated-sc-depth/06_04_2024_00:09:18/"

