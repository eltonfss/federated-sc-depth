import os
import copy
import time
import numpy as np
import torch
import math
import sys
from tqdm import tqdm
from pprint import pprint
from tensorboardX import SummaryWriter
from datetime import datetime
from pytorch_lightning import Trainer

from utils import set_seed, save_federated_training_state_json, save_args_json, average_weights, \
    load_weights_without_batchnorm, load_weights, compute_iid_sample_partitions
from configargs import get_configargs
from sc_depth_module_v3 import SCDepthModuleV3
from sc_depth_data_module import SCDepthDataModule

if __name__ == "__main__":

    # initialize federated training state
    # TODO read from JSON file
    federated_training_state = {}

    # parse config args
    config_args = get_configargs()
    federated_training_state['config_args'] = config_args
    device = "cuda" if torch.cuda.is_available() and config_args.gpu else "cpu"
    federated_training_state['device'] = device
    fed_train_num_rounds = config_args.fed_train_num_rounds
    fed_train_num_participants = config_args.fed_train_num_participants
    fed_train_frac_participants_per_round = config_args.fed_train_frac_participants_per_round
    fed_train_num_local_epochs = config_args.fed_train_num_local_epochs
    fed_train_num_local_train_batches = config_args.fed_train_num_local_train_batches
    fed_train_num_local_val_batches = config_args.fed_train_num_local_val_batches
    log_every_n_steps = config_args.log_every_n_steps
    fed_train_num_participants_per_round = max(int(fed_train_frac_participants_per_round * fed_train_num_participants), 1)
    load_weight_function = load_weights_without_batchnorm if config_args.fed_train_average_without_bn else load_weights
    fed_train_participant_order = config_args.fed_train_participant_order

    # set seed
    set_seed(config_args.seed)

    # setup logger
    federated_training_state['start_time'] = start_time = time.time()
    federated_training_state['start_datetime'] = start_datetime = datetime.now()
    federated_training_state['process_id'] = process_id = str(os.getpid())
    model_time = start_datetime.strftime("%d_%m_%Y_%H:%M:%S") + "_{}".format(process_id)
    federated_training_state['model_output_dir'] = model_output_dir = "save/" + model_time
    config_args.model_time = model_time
    save_args_json(model_output_dir, config_args)
    global_logger = SummaryWriter(os.path.join(model_output_dir, "tensorboard"))
    config_args.start_time = start_datetime

    # persist federated training state (Federation Checkpoint)
    save_federated_training_state_json(model_output_dir, federated_training_state)

    # print configuration info
    print("Pytorch CUDA is Available:", torch.cuda.is_available())
    print("PyTorch Device:", device)
    print("Output Model Directory:", model_output_dir)
    print("Number of Participants per Round: {}".format(fed_train_num_participants_per_round))
    print("Total Number of Rounds: {}".format(fed_train_num_rounds))
    pprint(config_args)

    # initialize global model
    print("Initializing Global Model ...")
    sc_depth_hparams = copy.deepcopy(config_args)
    sc_depth_hparams.batch_size = sc_depth_hparams.fed_train_local_batch_size
    sc_depth_hparams.lr = sc_depth_hparams.fed_train_local_learn_rate
    sc_depth_hparams.epoch_size = sc_depth_hparams.fed_train_num_local_train_batches
    global_model = None
    if config_args.model_version == 'v3':
        global_model = SCDepthModuleV3(sc_depth_hparams)
    if global_model is None:
        raise Exception("model_version is invalid! Only v3 is currently supported!")
    global_model = global_model.to(device)
    global_weights = global_model.state_dict()
    federated_training_state['initial_global_weights_bytesize'] = sys.getsizeof(global_weights)
    print("Global Model Initialized!")

    # persist federated training state (Federation Checkpoint)
    save_federated_training_state_json(model_output_dir, federated_training_state)

    # initialize local models
    print("Initializing Local Models ...")
    local_models = [copy.deepcopy(global_model) for _ in range(config_args.fed_train_num_participants)]
    print("Local Models Initialized!")

    # distribute training data
    print("Computing Dataset Distribution ...")
    global_data = SCDepthDataModule(sc_depth_hparams)
    global_data.setup()

    sample_train_indexes_by_participant = {}
    sample_val_indexes_by_participant = {}
    sample_test_indexes_by_participant = {}

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
            sample_train_indexes_by_participant[participant_index] = train_drive_sample_indexes

            # val indexes are always the same (every participant has the same val dataset)
            val_dataset_size = global_data.get_dataset_size("val")
            sample_val_indexes_by_participant[participant_index] = [i for i in range(val_dataset_size)]

            # test indexes are always the same (every participant has the same test dataset)
            test_dataset_size = global_data.get_dataset_size("test")
            sample_test_indexes_by_participant[participant_index] = [i for i in range(test_dataset_size)]

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
        sample_train_indexes_by_participant = compute_iid_sample_partitions(train_dataset_size,
                                                                            fed_train_num_participants)
        sample_val_indexes_by_participant = compute_iid_sample_partitions(val_dataset_size, fed_train_num_participants)
        sample_test_indexes_by_participant = compute_iid_sample_partitions(test_dataset_size,
                                                                           fed_train_num_participants)

    federated_training_state['sample_train_indexes_by_participant'] = sample_train_indexes_by_participant
    federated_training_state['sample_val_indexes_by_participant'] = sample_val_indexes_by_participant
    federated_training_state['sample_test_indexes_by_participant'] = sample_test_indexes_by_participant
    print("Dataset Distribution Computed!")

    # persist federated training state (Federation Checkpoint)
    save_federated_training_state_json(model_output_dir, federated_training_state)

    # run federated training
    print("Computing Federated Training ...")
    start_training_round = 0
    federated_training_state.update({
        'current_round': start_training_round,
        'max_rounds': fed_train_num_rounds,
        'fed_train_num_local_epochs': fed_train_num_local_epochs,
        'fed_train_num_local_train_batches': fed_train_num_local_train_batches,
        'fed_train_num_local_val_batches': fed_train_num_local_val_batches,
        'log_every_n_steps': log_every_n_steps,
        'total_training_time': None
    })
    local_update_time_by_round_by_participant = \
        federated_training_state['local_update_time_by_round_by_participant'] = {}
    global_update_time_by_round = federated_training_state['global_update_time_by_round'] = {}
    local_train_loss_by_round_by_participant = federated_training_state['local_train_loss_by_round_by_participant'] = {}
    local_val_loss_by_round_by_participant = federated_training_state['local_val_loss_by_round_by_participant'] = {}
    local_model_bytes_by_participant_by_round = \
        federated_training_state['local_model_bytes_by_participant_by_round'] = {}
    global_model_bytes_by_round = federated_training_state['global_model_bytes_by_round'] = {}
    num_participants_by_round = federated_training_state['num_participants_by_round'] = {}
    participant_order_by_round = federated_training_state['participant_order_by_round'] = {}

    # persist federated training state (Federation Checkpoint)
    save_federated_training_state_json(model_output_dir, federated_training_state)

    for training_round in tqdm(range(start_training_round, fed_train_num_rounds)):
        global_update_start_time = time.time()
        print(f"\n | Federated Training Round : {training_round} | Global Model : {model_time}\n")

        local_weights_by_participant = {}
        local_train_losses_by_participant = local_train_loss_by_round_by_participant[training_round]
        local_val_losses_by_participant = local_val_loss_by_round_by_participant[training_round]
        participants_ids = list(np.random.choice(range(fed_train_num_participants),
                                                 fed_train_num_participants_per_round, replace=False))
        num_participants_by_round[training_round] = len(participants_ids)
        local_update_time_by_participant = local_update_time_by_round_by_participant[training_round] = {}
        local_model_bytes_by_participant = local_model_bytes_by_participant_by_round[training_round] = {}

        # update each local model
        print("Computing Local Updates ...")
        if fed_train_participant_order == "sequential":
            participants_ids.sort()
            print("Local Updates of Participants will be computed in Sequential Order!")
        elif fed_train_participant_order == "random":
            print("Local Updates of Participants will be computed in Random Order!")
        else:
            raise Exception(f"Invalid Federated Training Participant Order: '{fed_train_participant_order}'! "
                            f"Only 'sequential' and 'random' are supported!")
        print(f"Participant Local Update Sequence is: {participants_ids}")
        participant_order_by_round[training_round] = participants_ids

        # persist federated training state (Federation Checkpoint)
        save_federated_training_state_json(model_output_dir, federated_training_state)

        for participant_id in participants_ids:
            local_update_start_time = time.time()
            federated_training_state['current_participant_id'] = participant_id

            # persist federated training state (Federation Checkpoint)
            save_federated_training_state_json(model_output_dir, federated_training_state)

            print(f"Computing Local Update of Participant {participant_id} ...")

            # configure data sampling for local training
            local_sample_train_indexes = sample_train_indexes_by_participant[participant_id]
            local_sample_val_indexes = sample_val_indexes_by_participant[participant_id]
            local_sample_test_indexes = sample_test_indexes_by_participant[participant_id]
            local_data = SCDepthDataModule(sc_depth_hparams,
                                           selected_train_sample_indexes=local_sample_train_indexes,
                                           selected_val_sample_indexes=local_sample_val_indexes,
                                           selected_test_sample_indexes=local_sample_test_indexes)

            # configure trainer for local training
            local_model = local_models[participant_id]
            trainer_config = dict(
                accelerator=device,
                log_every_n_steps=log_every_n_steps
            )
            if fed_train_num_local_epochs > 0:
                trainer_config['max_epochs'] = fed_train_num_local_epochs
            if fed_train_num_local_train_batches > 0:
                trainer_config['limit_train_batches'] = fed_train_num_local_train_batches
            if fed_train_num_local_val_batches > 0:
                trainer_config['limit_val_batches'] = fed_train_num_local_val_batches
            print("Local Trainer Config", trainer_config)
            local_trainer = Trainer(**trainer_config)

            # execute local training
            local_trainer.fit(local_model, local_data)

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
            else:
                # log local losses
                local_train_losses_by_participant[participant_id] = local_train_loss
                print(f"Local Train Loss of Participant {participant_id} in Round {training_round}: {local_train_loss}")
                local_val_losses_by_participant[participant_id] = local_val_loss
                print(f"Local Val Loss of Participant {participant_id} in Round {training_round}: {local_val_loss}")

                # log local model weights
                local_weights_of_participant = copy.deepcopy(local_model.state_dict())
                local_weights_by_participant[participant_id] = local_weights_of_participant
                local_model_bytes_of_participant = sys.getsizeof(local_weights_of_participant)
                local_model_bytes_by_participant[participant_id] = local_model_bytes_of_participant
                print(f"Local Model Bytes of Participant {participant_id} in Round {training_round}: "
                      f"{local_model_bytes_of_participant}")

                # TODO log local model checkpoint coordinates for restore purposes

                print(f"Local Update of Participant {participant_id} Computed!")

            # log local update time
            local_update_elapsed_time = time.time() - local_update_start_time
            local_update_time_by_participant[participant_id] = local_update_elapsed_time
            print("\n Local Update Took: {0:0.4f}".format(local_update_elapsed_time))

            # persist federated training state (Federation Checkpoint)
            save_federated_training_state_json(model_output_dir, federated_training_state)

        # log average local performance
        print("Local Updates Computed!")
        local_train_losses = list(local_train_losses_by_participant.values())
        avg_local_loss_train = sum(local_train_losses) / len(local_train_losses)
        local_val_losses = list(local_val_losses_by_participant.values())
        avg_local_loss_val = sum(local_val_losses) / len(local_val_losses)
        local_update_times = list(local_update_time_by_participant.values())
        avg_local_update_time = sum(local_update_times) / len(local_update_times)
        local_model_bytes = list(local_model_bytes_by_participant.values())
        avg_local_model_bytes = sum(local_model_bytes) / len(local_model_bytes)
        print(f"Average Local Training Loss in Round {training_round} = {avg_local_loss_train}")
        print(f"Average Local Validation Loss in Round {training_round} = {avg_local_loss_val}")
        print(f"Average Local Update Times in Round {training_round} = {avg_local_update_time}")
        print(f"Average Local Model Bytes in Round {training_round} = {avg_local_model_bytes}")

        # update global weights
        print(f"Computing Global Update ...")
        local_weights = list(local_weights_by_participant.values())
        global_weights = average_weights(local_weights)
        for i in range(fed_train_num_participants):
            local_models[i] = load_weight_function(local_models[i], global_weights)
        global_model.load_state_dict(global_weights)
        print(f"Global Update Computed!")

        # log global model weights
        global_model_bytes = sys.getsizeof(global_weights)
        global_model_bytes_by_round[training_round] = global_model_bytes
        print(f"Updated Global Model Bytes after Round {training_round}: {global_model_bytes}")

        # TODO log global model checkpoint coordinates for restore purposes

        # log global update time
        global_update_elapsed_time = time.time() - global_update_start_time
        global_update_time_by_round[training_round] = global_update_elapsed_time
        print("\n Global Update Took: {0:0.4f}".format(global_update_elapsed_time))
        
        # log total training time
        total_training_time = federated_training_state['total_training_time'] = time.time() - start_time
        print("\n Total Training Time so far: {0:0.4f}".format(total_training_time))

        # persist federated training state (Federation Checkpoint)
        save_federated_training_state_json(model_output_dir, federated_training_state)

    # log final total training time
    print("Federated Training Computed!")
    total_training_time = federated_training_state['total_training_time'] = time.time() - start_time
    print("\n Total Training Time: {0:0.4f}".format(total_training_time))

    # persist federated training state (Federation Checkpoint)
    save_federated_training_state_json(model_output_dir, federated_training_state)
