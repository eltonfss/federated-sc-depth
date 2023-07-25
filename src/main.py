import os
import copy
import time
import traceback
import gc

import numpy as np
from types import SimpleNamespace
import torch
import math
import sys
from tqdm import tqdm
from pprint import pprint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime
from pytorch_lightning import Trainer

from utils import set_seed, restore_federated_training_state, backup_federated_training_state, estimate_model_size, \
    average_weights_by_num_samples, average_weights_with_loss_optimization, load_weights_without_batchnorm, load_weights, compute_iid_sample_partitions, \
    mkdir_if_missing
from configargs import get_configargs
from sc_depth_module_v3 import SCDepthModuleV3
from sc_depth_data_module import SCDepthDataModule

if __name__ == "__main__":
    config_args = get_configargs()
    restore_dir = config_args.fed_train_state_restore_dir
    backup_dir = config_args.fed_train_state_backup_dir

    # initialize federated training state (backup/restore)
    federated_training_state = {}
    restoring_federation_state = False
    model_save_dir = None
    model_time = None
    start_time = None
    start_datetime = None
    device = None
    global_model_weights_filename = "global_model_weights"
    if restore_dir:
        if not os.path.exists(restore_dir):
            raise Exception(f"Restore Directory Informed ({restore_dir}) does not exist!")
        model_save_dir = restore_dir
        restoring_federation_state = True
        federated_training_state = restore_federated_training_state(model_save_dir)
        model_time = federated_training_state['model_time']
        start_time = float(federated_training_state['start_time'])
        start_datetime = federated_training_state['start_datetime']
        config_args = SimpleNamespace(**federated_training_state['config_args'])
        device = federated_training_state['device']
    else:
        start_time = time.time()
        federated_training_state['start_time'] = str(start_time)
        start_datetime = datetime.now()
        federated_training_state['start_datetime'] = str(start_datetime)
        model_time = start_datetime.strftime("%d_%m_%Y_%H:%M:%S")
        federated_training_state['model_time'] = model_time
        mkdir_if_missing(backup_dir)
        federated_training_state['model_save_dir'] = model_save_dir = os.path.join(backup_dir, model_time)
        mkdir_if_missing(model_save_dir)
        federated_training_state['config_args'] = vars(config_args)
        device = "cuda" if torch.cuda.is_available() and config_args.gpu else "cpu"
        federated_training_state['device'] = device
    config_args.model_time = model_time
    config_args.start_time = str(start_datetime)
    federated_training_state["training_paused"] = False
    skip_restore = federated_training_state.get("skip_restore", False)

    # parse config args
    fed_train_num_rounds = config_args.fed_train_num_rounds
    fed_train_num_participants = config_args.fed_train_num_participants
    fed_train_frac_participants_per_round = config_args.fed_train_frac_participants_per_round
    fed_train_num_local_epochs = config_args.fed_train_num_local_epochs
    fed_train_num_local_train_batches = config_args.fed_train_num_local_train_batches
    fed_train_num_local_val_batches = config_args.fed_train_num_local_val_batches
    log_every_n_steps = config_args.log_every_n_steps
    fed_train_num_participants_per_round = max(
        math.ceil(fed_train_frac_participants_per_round * fed_train_num_participants), 1
    )
    load_weight_function = load_weights_without_batchnorm if config_args.fed_train_average_without_bn else load_weights
    fed_train_participant_order = config_args.fed_train_participant_order
    fed_train_num_local_sanity_val_steps = config_args.fed_train_num_local_sanity_val_steps
    fed_train_average_grid_optimization_range = config_args.fed_train_average_grid_optimization_range

    # set seed
    set_seed(config_args.seed)

    # persist federated training state (Federation Checkpoint)
    backup_federated_training_state(model_save_dir, federated_training_state)

    # print configuration info
    print("Pytorch CUDA is Available:", torch.cuda.is_available())
    print("PyTorch Device:", device)
    print("Model Save Directory:", model_save_dir)
    print("Number of Participants per Round: {}".format(fed_train_num_participants_per_round))
    print("Total Number of Rounds: {}".format(fed_train_num_rounds))
    pprint(config_args)

    # initialize global model
    print("Initializing Global Model ...")
    sc_depth_hparams = copy.deepcopy(config_args)
    batch_size = sc_depth_hparams.batch_size = sc_depth_hparams.fed_train_local_batch_size
    sc_depth_hparams.lr = sc_depth_hparams.fed_train_local_learn_rate
    sc_depth_hparams.epoch_size = sc_depth_hparams.fed_train_num_local_train_batches
    global_model = None
    global_model_round = None
    if config_args.model_version == 'v3':
        global_model = SCDepthModuleV3(sc_depth_hparams)
    if global_model is None:
        raise Exception("model_version is invalid! Only v3 is currently supported!")
    if restoring_federation_state and not skip_restore:
        global_model_round = int(federated_training_state["current_round"])
        print(f"Trying to restore Global Model Weights from Round {global_model_round} ...")
        round_model_dir = os.path.join(model_save_dir, f'round_{global_model_round}')
        global_model_weights_filepath = os.path.join(round_model_dir, f"{global_model_weights_filename}.pt")
        while not os.path.exists(global_model_weights_filepath) and global_model_round > 0:
            previous_round = global_model_round - 1
            print(
                f"Global Weights from Round {global_model_round} were not saved, restoring from Round {previous_round}...")
            round_model_dir = os.path.join(model_save_dir, f'round_{previous_round}')
            global_model_weights_filepath = os.path.join(round_model_dir, f"{global_model_weights_filename}.pt")
            global_model_round = previous_round
        try:
            print(f'loading pre-trained global model weights from {global_model_weights_filepath} ...')
            global_weights = torch.load(global_model_weights_filepath)
            global_model.load_state_dict(global_weights)
            global_model.eval()
        except:
            traceback.print_exc()
            print(f"WARNING: Could not load pre-trained global model weights from {round_model_dir}!"
                  f"Will proceed with default global model initialization.")
            global_model_round = None
    global_model = global_model.to(device)
    global_weights = global_model.state_dict()
    previous_global_model = copy.deepcopy(global_model)
    if federated_training_state.get('initial_global_weights_bytesize', None) is None:
        federated_training_state['initial_global_weights_bytesize'] = estimate_model_size(global_weights)
    print("Global Model Initialized!")

    # persist federated training state (Federation Checkpoint)
    backup_federated_training_state(model_save_dir, federated_training_state)

    # distribute training data
    print("Setting Up Training Dataset ...")
    global_data = SCDepthDataModule(sc_depth_hparams)
    global_data.setup()
    print("Training Dataset Setup Completed!")

    sample_train_indexes_by_participant = federated_training_state.get('sample_train_indexes_by_participant', {})
    sample_val_indexes_by_participant = federated_training_state.get('sample_val_indexes_by_participant', {})
    sample_test_indexes_by_participant = federated_training_state.get('sample_test_indexes_by_participant', {})

    if not restoring_federation_state:
        print("Computing Dataset distribution across participants ...")
        if config_args.fed_train_by_drive:
            train_drive_ids = global_data.get_drive_ids("train")
            print(len(train_drive_ids), "TRAIN DRIVE IDS FOUND")

            # validate number of participants
            if fed_train_num_participants > len(train_drive_ids):
                raise Exception("fed_train_num_participants cannot be greater the number of train drive ids available!")

            # distribute sample indexes by participant
            for participant_index, drive_id in enumerate(train_drive_ids):
                # train indexes vary based on drive id
                train_drive_samples = global_data.get_samples_by_drive_id("train", drive_id)
                print(len(train_drive_samples), "TRAIN SAMPLES FOUND FOR DRIVE ID", drive_id)
                train_drive_sample_indexes = [sample['sourceIndex'] for sample in train_drive_samples]
                sample_train_indexes_by_participant[str(participant_index)] = train_drive_sample_indexes

                # val indexes are always the same (every participant has the same val dataset)
                val_dataset_size = global_data.get_dataset_size("val")
                sample_val_indexes_by_participant[str(participant_index)] = [i for i in range(val_dataset_size)]

                # test indexes are always the same (every participant has the same test dataset)
                test_dataset_size = global_data.get_dataset_size("test")
                sample_test_indexes_by_participant[str(participant_index)] = [i for i in range(test_dataset_size)]

            if config_args.fed_train_by_drive_sort == 'eager':
                # order based on the number of samples
                sorted_sample_train_indexes_by_participant = sorted(sample_train_indexes_by_participant.items(),
                                                                    key=lambda x: len(x[1]), reverse=True)
                for new_participant_index in range(len(sorted_sample_train_indexes_by_participant)):
                    key_value_pair = sorted_sample_train_indexes_by_participant[new_participant_index]
                    old_participant_index, participant_train_indexes = key_value_pair
                    participant_val_indexes = sample_val_indexes_by_participant[old_participant_index]
                    participant_test_indexes = sample_test_indexes_by_participant[old_participant_index]
                    new_participant_index = str(new_participant_index)
                    sample_train_indexes_by_participant[new_participant_index] = participant_train_indexes
                    sample_val_indexes_by_participant[new_participant_index] = participant_val_indexes
                    sample_test_indexes_by_participant[new_participant_index] = participant_test_indexes

                if fed_train_num_participants < len(train_drive_ids) and \
                        config_args.fed_train_by_drive_redistribute_remaining:

                    # get indexes left out
                    sample_train_indexes_left_out = []
                    for participant_index, sample_train_indexes in sample_train_indexes_by_participant.items():
                        if int(participant_index) < fed_train_num_participants:
                            continue
                        sample_train_indexes_left_out.extend(sample_train_indexes)
                        sample_train_indexes_by_participant[participant_index] = []

                    # sort them
                    sample_train_indexes_left_out = sorted(sample_train_indexes_left_out)

                    # randomize them
                    sample_train_indexes_left_out = list(np.random.choice(
                        sample_train_indexes_left_out, len(sample_train_indexes_left_out), replace=False)
                    )
                    sample_train_indexes_left_out = [int(i) for i in sample_train_indexes_left_out]

                    # partition by participants
                    number_of_samples_left_out = len(sample_train_indexes_left_out)
                    partitions_of_sample_train_indexes_left_out = []
                    partition_size = math.ceil(number_of_samples_left_out/fed_train_num_participants)
                    for i in range(0, number_of_samples_left_out, partition_size):
                        partitions_of_sample_train_indexes_left_out.append(
                            sample_train_indexes_left_out[i:i+partition_size]
                        )

                    # extend participant indexes list with partition of indexes left out
                    for participant_index, sample_train_indexes in sample_train_indexes_by_participant.items():
                        if int(participant_index) >= fed_train_num_participants:
                            break
                        participant_partition = partitions_of_sample_train_indexes_left_out[int(participant_index)]
                        sample_train_indexes_by_participant[participant_index].extend(participant_partition)

            elif config_args.fed_train_by_drive_sort == 'random':
                raise NotImplementedError

            # check for duplicates
            intersection_set = set()
            list_of_sets = [set(value) for value in sample_train_indexes_by_participant.values()]
            for i, value_set_a in enumerate(list_of_sets):
                for j, value_set_b in enumerate(list_of_sets):
                    if i == j:
                        continue
                    intersection = value_set_a.intersection(value_set_b)
                    intersection_set = intersection_set.union(intersection)
            if len(intersection_set) > 0:
                print(len(intersection_set), "duplicates found!")
                raise Exception("Dataset Distribution Error!")
        else:
            train_dataset_size = global_data.get_dataset_size("train")
            val_dataset_size = global_data.get_dataset_size("val")
            test_dataset_size = global_data.get_dataset_size("test")

            # distribute sample indexes by participant
            sample_train_indexes_by_participant = compute_iid_sample_partitions(
                dataset_size=train_dataset_size, num_partitions=fed_train_num_participants
            )
            for participant_index in range(fed_train_num_participants):
                # val indexes are always the same (every participant has the same val dataset)
                val_dataset_size = global_data.get_dataset_size("val")
                sample_val_indexes_by_participant[str(participant_index)] = [i for i in range(val_dataset_size)]
                # test indexes are always the same (every participant has the same test dataset)
                test_dataset_size = global_data.get_dataset_size("test")
                sample_test_indexes_by_participant[str(participant_index)] = [i for i in range(test_dataset_size)]

        federated_training_state['sample_train_indexes_by_participant'] = sample_train_indexes_by_participant
        federated_training_state['sample_val_indexes_by_participant'] = sample_val_indexes_by_participant
        federated_training_state['sample_test_indexes_by_participant'] = sample_test_indexes_by_participant
        print("Dataset Distribution Computed!")
    else:
        print("Dataset distribution restored from federated training state backup!")

    # persist federated training state (Federation Checkpoint)
    backup_federated_training_state(model_save_dir, federated_training_state)

    try:
        # run federated training
        print("Computing Federated Training ...")
        start_training_round = federated_training_state.get('current_round', 0)
        if not restoring_federation_state:
            federated_training_state.update({
                'current_round': start_training_round,
                'max_rounds': fed_train_num_rounds,
                'fed_train_num_local_epochs': fed_train_num_local_epochs,
                'fed_train_num_local_train_batches': fed_train_num_local_train_batches,
                'fed_train_num_local_val_batches': fed_train_num_local_val_batches,
                'log_every_n_steps': log_every_n_steps,
                'total_training_time': None
            })
            federated_training_state['local_update_time_by_round_by_participant'] = {}
            federated_training_state['global_update_time_by_round'] = {}
            federated_training_state['local_train_loss_by_round_by_participant'] = {}
            federated_training_state['local_val_loss_by_round_by_participant'] = {}
            federated_training_state['local_model_dir_by_participant_by_round'] = {}
            federated_training_state['local_model_bytes_by_participant_by_round'] = {}
            federated_training_state['global_model_bytes_by_round'] = {}
            federated_training_state['global_model_weights_of_weights_by_round'] = {}
            federated_training_state['global_model_standard_fed_avg_by_round'] = {}
            federated_training_state['global_model_dir_by_round'] = {}
            federated_training_state['num_participants_by_round'] = {}
            federated_training_state['participant_order_by_round'] = {}
            federated_training_state['global_test_loss_by_round'] = {}
        local_update_time_by_round_by_participant = federated_training_state[
            'local_update_time_by_round_by_participant']
        global_update_time_by_round = federated_training_state['global_update_time_by_round']
        local_train_loss_by_round_by_participant = federated_training_state['local_train_loss_by_round_by_participant']
        local_val_loss_by_round_by_participant = federated_training_state['local_val_loss_by_round_by_participant']
        local_model_dir_by_participant_by_round = federated_training_state['local_model_dir_by_participant_by_round']
        local_model_bytes_by_participant_by_round = federated_training_state[
            'local_model_bytes_by_participant_by_round']
        global_model_bytes_by_round = federated_training_state['global_model_bytes_by_round']
        global_model_weights_of_weights_by_round = federated_training_state['global_model_weights_of_weights_by_round']
        global_model_standard_fed_avg_by_round = federated_training_state['global_model_standard_fed_avg_by_round']
        global_model_dir_by_round = federated_training_state['global_model_dir_by_round']
        num_participants_by_round = federated_training_state['num_participants_by_round']
        participant_order_by_round = federated_training_state['participant_order_by_round']
        global_test_loss_by_round = federated_training_state['global_test_loss_by_round']

        # persist federated training state (Federation Checkpoint)
        backup_federated_training_state(model_save_dir, federated_training_state)

        # compute participant order
        num_participants = fed_train_num_participants_per_round
        participants_ids = federated_training_state.get('ordered_participant_ids', [])
        if len(participants_ids) < num_participants:
            participants_ids = [int(participant_id) for participant_id in range(fed_train_num_participants)]
            if fed_train_participant_order == 'sequential':
                print("Local Updates of Participants will be computed in Sequential Order!")
            elif fed_train_participant_order == "random":
                participants_ids = list(np.random.choice(participants_ids, len(participants_ids), replace=False))
                participants_ids = [int(participant_id) for participant_id in participants_ids]
                print("Local Updates of Participants will be computed in Random Order!")
            else:
                raise Exception(f"Invalid Federated Training Participant Order: '{fed_train_participant_order}'! "
                                f"Only 'sequential' and 'random' are supported!")
        else:
            print("Ordered Participant IDS restored from federated training state!")
        federated_training_state['ordered_participant_ids'] = participants_ids
        print("Ordered Participant IDs:", participants_ids)

        for training_round in tqdm(range(int(start_training_round), int(fed_train_num_rounds))):
            training_round = str(training_round)
            federated_training_state['current_round'] = training_round
            global_update_start_time = time.time()
            print(f"\n | Federated Training Round : {training_round} | Global Model : {model_time}\n")

            local_weights_by_participant = {}
            num_train_samples_by_participant = {}
            local_train_losses_by_participant = local_train_loss_by_round_by_participant.get(training_round, {})
            local_train_loss_by_round_by_participant[training_round] = local_train_losses_by_participant
            local_val_losses_by_participant = local_val_loss_by_round_by_participant.get(training_round, {})
            local_val_loss_by_round_by_participant[training_round] = local_val_losses_by_participant
            round_participants_ids = participant_order_by_round.get(training_round, participants_ids)
            if not restoring_federation_state:
                number_of_partitions = math.ceil(fed_train_num_participants / num_participants)
                current_partition = int(training_round)
                if current_partition > number_of_partitions - 1:
                    current_partition = (int(training_round)) % number_of_partitions
                start_index = current_partition * num_participants
                max_index = len(round_participants_ids)
                next_partition = current_partition + 1
                end_index = min(start_index+num_participants, max_index)
                round_participants_ids = round_participants_ids[start_index:end_index]
                num_participants_by_round[training_round] = len(round_participants_ids)
                print(f"Participant Local Update Sequence is: {round_participants_ids}")
                participant_order_by_round[training_round] = round_participants_ids
            local_update_time_by_participant = local_update_time_by_round_by_participant.get(training_round, {})
            local_update_time_by_round_by_participant[training_round] = local_update_time_by_participant
            local_model_bytes_by_participant = local_model_bytes_by_participant_by_round.get(training_round, {})
            local_model_bytes_by_participant_by_round[training_round] = local_model_bytes_by_participant
            local_model_dir_by_participant = local_model_dir_by_participant_by_round.get(training_round, {})
            local_model_dir_by_participant_by_round[training_round] = local_model_dir_by_participant

            round_dir = f'round_{training_round}'
            round_model_dir = os.path.join(model_save_dir, round_dir)
            mkdir_if_missing(round_model_dir)
            global_logger = TensorBoardLogger(save_dir=model_save_dir, name=round_dir, version=0)
            global_checkpoint_path = os.path.join(round_model_dir, "last.ckpt")
            global_checkpoint_callback = ModelCheckpoint(dirpath=round_model_dir,
                                                         save_last=True,
                                                         save_weights_only=False,
                                                         monitor='test_loss',
                                                         mode='min',
                                                         verbose=True,
                                                         save_top_k=3)
            global_trainer_config = dict(
                accelerator=device,
                log_every_n_steps=log_every_n_steps,
                callbacks=[global_checkpoint_callback],
                logger=global_logger,
                benchmark=True
            )

            # persist federated training state (Federation Checkpoint)
            backup_federated_training_state(model_save_dir, federated_training_state)

            skip_local_updates = global_model_round is not None and int(global_model_round) >= int(training_round)
            if not skip_local_updates or skip_restore:

                # update each local model
                print("\nComputing Local Updates ...")
                for participant_id in round_participants_ids:

                    print(f"Computing Local Update of Participant {participant_id} ...")
                    local_update_start_time = time.time()

                    # configure data sampling for local training
                    local_sample_train_indexes = sample_train_indexes_by_participant[str(participant_id)]
                    num_train_samples_by_participant[participant_id] = len(local_sample_train_indexes)
                    local_sample_val_indexes = sample_val_indexes_by_participant[str(participant_id)]
                    local_sample_test_indexes = sample_test_indexes_by_participant[str(participant_id)]
                    local_data = SCDepthDataModule(sc_depth_hparams,
                                                   selected_train_sample_indexes=local_sample_train_indexes,
                                                   selected_val_sample_indexes=local_sample_val_indexes,
                                                   selected_test_sample_indexes=local_sample_test_indexes)

                    # prepare local model for update
                    local_model = copy.deepcopy(global_model)
                    participant_dir = f'participant_{participant_id}'
                    participant_model_dir = os.path.join(round_model_dir, participant_dir)
                    local_checkpoint_path = os.path.join(participant_model_dir, "last.ckpt")
                    if restoring_federation_state:
                        if not skip_restore:
                            try:
                                print('load pre-trained model from {}'.format(local_checkpoint_path))
                                local_model = local_model.load_from_checkpoint(
                                    checkpoint_path=local_checkpoint_path,
                                    strict=False,
                                    hparams=sc_depth_hparams
                                )
                                local_weights_of_participant = copy.deepcopy(local_model.state_dict())
                                local_weights_by_participant[participant_id] = local_weights_of_participant
                                local_model_bytes_of_participant = estimate_model_size(local_weights_of_participant)
                                local_model_bytes_by_participant[participant_id] = local_model_bytes_of_participant
                                if participant_id != federated_training_state['current_participant_id']:
                                    continue
                                else:
                                    restoring_federation_state = False
                            except:
                                traceback.print_exc()
                                print(f"WARNING: Could not load local model checkpoint from {local_checkpoint_path}!"
                                      f"Will proceed with local model version initialized from global model.")
                        else:
                            print("Skipping Restore ...")
                            federated_training_state['skip_restore'] = skip_restore = False
                            restoring_federation_state = False

                    federated_training_state['current_participant_id'] = participant_id

                    # persist federated training state (Federation Checkpoint)
                    backup_federated_training_state(model_save_dir, federated_training_state)

                    # configure logger for local training
                    local_logger = TensorBoardLogger(save_dir=round_model_dir, name=participant_dir, version=0)
                    n_train_steps_to_save = int(max(25, fed_train_num_local_train_batches / batch_size))
                    local_checkpoint_callback = ModelCheckpoint(dirpath=participant_model_dir,
                                                                save_last=True,
                                                                save_weights_only=False,
                                                                every_n_train_steps=n_train_steps_to_save,
                                                                save_on_train_epoch_end=True,
                                                                verbose=True)

                    # configure trainer for local training
                    trainer_config = dict(
                        accelerator=device,
                        log_every_n_steps=log_every_n_steps,
                        num_sanity_val_steps=fed_train_num_local_sanity_val_steps,
                        callbacks=[local_checkpoint_callback],
                        logger=local_logger,
                        benchmark=True
                    )
                    if fed_train_num_local_epochs > 0:
                        trainer_config['max_epochs'] = fed_train_num_local_epochs
                    if fed_train_num_local_train_batches > 0:
                        trainer_config['limit_train_batches'] = min(
                            math.floor(num_train_samples_by_participant[participant_id]/batch_size),
                            fed_train_num_local_train_batches
                        )
                    if fed_train_num_local_val_batches > 0:
                        trainer_config['limit_val_batches'] = fed_train_num_local_val_batches
                    print("Local Trainer Config", trainer_config)
                    local_trainer = Trainer(**trainer_config)

                    # execute local training
                    fit_config = dict(model=local_model, datamodule=local_data)
                    if os.path.exists(local_checkpoint_path):
                        fit_config.update(dict(ckpt_path=local_checkpoint_path))
                    try:
                        local_trainer.fit(**fit_config)
                    except RuntimeError as exception:
                        if "out of memory" in str(exception):
                            print(torch.cuda.list_gpu_processes())
                            gc.collect()
                            torch.cuda.empty_cache()
                        raise exception
                    if local_trainer.interrupted:
                        raise KeyboardInterrupt("Local Update Interrupted")

                    # validate local training
                    train_epoch_losses = local_model.train_epoch_losses
                    final_local_loss_train = train_epoch_losses[-1] if len(train_epoch_losses) > 0 else None
                    val_epoch_losses = local_model.val_epoch_losses
                    final_local_loss_val = val_epoch_losses[-1] if len(val_epoch_losses) > 0 else None
                    local_train_loss = copy.deepcopy(final_local_loss_train)
                    ignore = False
                    if local_train_loss is None or math.isnan(local_train_loss) or math.isinf(local_train_loss):
                        print(f"Local Train Loss of Participant {participant_id} in Round {training_round} is invalid!")
                        ignore = True
                    local_val_loss = copy.deepcopy(final_local_loss_val)
                    if local_val_loss is None or math.isnan(local_val_loss) or math.isinf(local_val_loss):
                        print(f"Local Val Loss of Participant {participant_id} in Round {training_round} is invalid!")
                        ignore = True
                    if ignore:
                        print(f"Ignoring Local Update of Participant {participant_id} in Round {training_round}")
                        # local_val_losses_by_participant[participant_id] = 99999999999999999999999 # sudo infinite loss
                    else:
                        # log local losses
                        local_train_losses_by_participant[participant_id] = local_train_loss.tolist()
                        print(
                            f"Local Train Loss of Participant {participant_id} in Round {training_round}: {local_train_loss}")
                        local_val_losses_by_participant[participant_id] = local_val_loss
                        print(
                            f"Local Val Loss of Participant {participant_id} in Round {training_round}: {local_val_loss}")

                        # log local model weights
                        local_weights_of_participant = copy.deepcopy(local_model.state_dict())
                        local_weights_by_participant[participant_id] = local_weights_of_participant
                        local_model_bytes_of_participant = estimate_model_size(local_weights_of_participant)
                        local_model_bytes_by_participant[participant_id] = local_model_bytes_of_participant
                        print(f"Local Model Size of Participant {participant_id} in Round {training_round}: "
                              f"{local_model_bytes_of_participant} Bytes")

                        # log local model checkpoint coordinates for restore purposes
                        local_model_dir_by_participant[participant_id] = participant_model_dir
                        print(
                            f"Local Model of Participant {participant_id} in Round {training_round} has been saved at:",
                            participant_model_dir)

                        print(f"Local Update of Participant {participant_id} Computed!")

                    # log local update time
                    local_update_elapsed_time = time.time() - local_update_start_time
                    local_update_time_by_participant[participant_id] = float(local_update_elapsed_time)
                    print("Local Update Took: {0:0.4f} seconds".format(local_update_elapsed_time))

                    # persist federated training state (Federation Checkpoint)
                    backup_federated_training_state(model_save_dir, federated_training_state)

                # log average local performance
                print("Local Updates Computed!")
                local_train_losses = list(local_train_losses_by_participant.values())
                if len(local_train_losses) > 0:
                    avg_local_loss_train = sum(local_train_losses) / len(local_train_losses)
                    print(f"Average Local Training Loss in Round {training_round} = {avg_local_loss_train}")
                local_val_losses = list(local_val_losses_by_participant.values())
                if len(local_val_losses) > 0:
                    avg_local_loss_val = sum(local_val_losses) / len(local_val_losses)
                    print(f"Average Local Validation Loss in Round {training_round} = {avg_local_loss_val}")
                local_update_times = list(local_update_time_by_participant.values())
                if len(local_update_times) > 0:
                    avg_local_update_time = sum(local_update_times) / len(local_update_times)
                    print(f"Average Local Update Times in Round {training_round} = {avg_local_update_time}")
                local_model_bytes = list(local_model_bytes_by_participant.values())
                if len(local_model_bytes) > 0:
                    avg_local_model_bytes = sum(local_model_bytes) / len(local_model_bytes)
                    print(f"Average Local Model Size in Round {training_round} = {avg_local_model_bytes} Bytes")

                # update global weights
                ordered_local_weights = list(local_weights_by_participant.values())
                ordered_num_train_samples = list(num_train_samples_by_participant.values())
                if len(ordered_local_weights) > 0:
                    print(f"Computing Global Update ...")
                    weights_of_weights = None
                    standard_fed_avg = True
                    if fed_train_average_grid_optimization_range == -1:
                        global_weights, weights_of_weights = average_weights_by_num_samples(
                            ordered_local_weights, ordered_num_train_samples
                        )
                    else:
                        global_weights, weights_of_weights, standard_fed_avg = average_weights_with_loss_optimization(
                            ordered_local_weights, ordered_num_train_samples,
                            global_model, global_data, global_trainer_config,
                            fed_train_average_grid_optimization_range
                        )
                    global_model_weights_of_weights_by_round[training_round] = weights_of_weights
                    global_model_standard_fed_avg_by_round[training_round] = standard_fed_avg
                    previous_global_model = global_model
                    global_model.load_state_dict(global_weights)
                    print(f"Global Update Computed!")
                else:
                    print("WARNING: Not enough local updates were successful! Global Model will remain the same!")
            elif restoring_federation_state:
                print("Skipping Local Updates ...")
                restoring_federation_state = False
                global_model_round = None

            print("Testing Global Model after Update...")
            global_trainer = Trainer(**global_trainer_config)
            test_config = dict(model=global_model, datamodule=global_data)
            if skip_local_updates and os.path.exists(global_checkpoint_path):
                test_config.update(dict(ckpt_path=global_checkpoint_path))
            global_trainer.test(**test_config)
            if global_trainer.interrupted:
                raise KeyboardInterrupt("Global Update Interrupted!")
            test_epoch_losses = global_model.test_epoch_losses
            test_epoch_loss = test_epoch_losses[-1] if len(test_epoch_losses) > 0 else None
            if int(training_round) > 0:
                previous_round = str(int(training_round) - 1)
                previous_test_epoch_loss = float(global_test_loss_by_round[previous_round])
                if previous_test_epoch_loss < test_epoch_loss and previous_global_model is not None:
                    print("Updated Global Model has worse test loss than the previous one! ")
                    print(f"Replacing it with the global model from round {previous_round}...")
                    print(f"Test loss of round {training_round} will remain the same of round {previous_round}: {previous_test_epoch_loss}")
                    test_epoch_loss = previous_test_epoch_loss
                    global_model = previous_global_model
                    global_weights = global_model.state_dict()

            global_test_loss_by_round[training_round] = test_epoch_loss
            print(f"Global Test Loss in Round {training_round}: {test_epoch_loss}")

            print(f"Saving Global Model Weights at {round_model_dir} ...")
            global_model_weights_filepath = os.path.join(round_model_dir, f"{global_model_weights_filename}.pt")
            torch.save(global_weights, global_model_weights_filepath)
            print(f"Global Model Weights saved as : {global_model_weights_filepath}")

            # log global model weights
            global_model_bytes = estimate_model_size(global_weights)
            global_model_bytes_by_round[training_round] = global_model_bytes
            print(f"Updated Global Model Size after Round {training_round}: {global_model_bytes} Bytes")

            # log global model checkpoint coordinates for restore purposes
            global_model_dir_by_round[training_round] = round_model_dir
            print(f"Global Model of Round {training_round} has been saved at:", round_model_dir)

            # log global update time
            global_update_elapsed_time = time.time() - global_update_start_time
            global_update_time_by_round[training_round] = float(global_update_elapsed_time)
            print("Global Update Took: {0:0.4f} seconds".format(global_update_elapsed_time))

            # log total training time
            total_training_time = time.time() - start_time
            federated_training_state['total_training_time'] = float(total_training_time)
            print("Total Training Time so far: {0:0.4f} seconds".format(total_training_time))

            # persist federated training state (Federation Checkpoint)
            backup_federated_training_state(model_save_dir, federated_training_state)
            restoring_federation_state = False

        # log final total training time
        print("Federated Training Computed!")
        total_training_time = time.time() - start_time
        federated_training_state['total_training_time'] = float(total_training_time)
        print("Total Training Time: {0:0.4f} seconds".format(total_training_time))

        # log end time
        end_time = time.time()
        federated_training_state['end_time'] = str(end_time)
        end_datetime = datetime.now()
        federated_training_state['end_datetime'] = str(end_datetime)
        print("Training Finished at:", end_datetime)

        # persist federated training state (Federation Checkpoint)
        backup_federated_training_state(model_save_dir, federated_training_state)
    except KeyboardInterrupt as exception:
        print("\nFederated Training interruption request detected!")
        print("Saving Federated Training State and stopping training ...")
        federated_training_state["training_paused"] = True
        backup_federated_training_state(model_save_dir, federated_training_state)
    finally:
        print("Federated Training State Saved at:", model_save_dir)
