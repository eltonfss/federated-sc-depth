import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

import os
import copy
import time
import numpy as np
import torch
from tqdm import tqdm
from pprint import pprint
from tensorboardX import SummaryWriter
from datetime import datetime
from pytorch_lightning import Trainer

from utils import set_seed, mkdir_if_missing, save_args_json, average_weights, load_weights_without_batchnorm, load_weights, compute_iid_sample_partitions
from configargs import get_configargs
from sc_depth_module_v1 import SCDepthModuleV1
from sc_depth_module_v2 import SCDepthModuleV2
from sc_depth_module_v3 import SCDepthModuleV3
from sc_depth_data_module import SCDepthDataModule

if __name__ == "__main__":
    
    # parse configargs
    config_args = get_configargs()
    device = "cuda" if torch.cuda.is_available() and config_args.gpu else "cpu"
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
    start_time = time.time()
    start_datetime = datetime.now()
    process_id = str(os.getpid())
    model_time = start_datetime.strftime("%d_%m_%Y_%H:%M:%S") + "_{}".format(process_id)
    model_output_dir = "save/" + model_time
    config_args.model_time = model_time
    save_args_json(model_output_dir, config_args)
    logger = SummaryWriter(model_output_dir + "/tensorboard")
    config_args.start_time = start_datetime
    
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
    if config_args.model_version == 'v1':
        global_model = SCDepthModuleV1(sc_depth_hparams)
    elif config_args.model_version == 'v2':
        global_model = SCDepthModuleV2(sc_depth_hparams)
    elif config_args.model_version == 'v3':
        global_model = SCDepthModuleV3(sc_depth_hparams)  
    global_model = global_model.to(device)
    global_weights = global_model.state_dict()
    print("Global Model Initialized!")

    # initialize local models
    print("Initializing Local Models ...")
    local_models = [copy.deepcopy(global_model) for _ in range(config_args.fed_train_num_participants)]
    print("Local Models Initialized!")
    
    # distribute training data
    print("Computing Dataset Distribution ...")
    global_data = SCDepthDataModule(sc_depth_hparams)
    global_data.setup()

    if config_args.fed_train_by_drive:
        train_drive_ids = global_data.get_drive_ids("train")
        print(len(train_drive_ids), "TRAIN DRIVE IDS FOUND") 
        
        # validate number of participants
        if fed_train_num_participants > len(train_drive_ids):
            raise "fed_train_num_participants cannot be greather than the number of train drive ids availabe in the dataset!"
        
        sample_train_indexes_by_participant = {}
        sample_val_indexes_by_participant = {}
        sample_test_indexes_by_participant = {}
        for participant_index, drive_id in enumerate(train_drive_ids):

            # train
            train_drive_samples = global_data.get_samples_by_drive_id("train", drive_id)
            print(len(train_drive_samples), "TRAIN SAMPLES FOUND FOR DRIVE ID", drive_id)
            train_drive_sample_indexes = [sample['sourceIndex'] for sample in train_drive_samples]
            sample_train_indexes_by_participant[participant_index] = train_drive_sample_indexes

            # val
            val_dataset_size = global_data.get_dataset_size("val")
            sample_val_indexes_by_participant[participant_index] = [i for i in range(val_dataset_size)]

            # test
            test_dataset_size = global_data.get_dataset_size("test")
            sample_test_indexes_by_participant[participant_index] = [i for i in range(test_dataset_size)]

        # check for duplicates
        intersection_set = set()
        list_of_sets = [set(value) for value in sample_train_indexes_by_participant.values()]
        for i, value_set_a in enumerate(list_of_sets): 
            for j, value_set_b in enumerate(list_of_sets): 
                if i == j: continue           
                intersection = value_set_a.intersection(value_set_b)
                intersection_set = intersection_set.union(intersection)
        if len(intersection_set) > 0:
            print(len(intersection_set), "duplicates found!")
            raise "Dataset Distribution Error!"
        
    else:
        train_dataset_size = global_data.get_dataset_size("train")
        val_dataset_size = global_data.get_dataset_size("val")
        test_dataset_size = global_data.get_dataset_size("test")
        sample_train_indexes_by_participant = compute_iid_sample_partitions(train_dataset_size, fed_train_num_participants)
        sample_val_indexes_by_participant = compute_iid_sample_partitions(val_dataset_size, fed_train_num_participants)
        sample_test_indexes_by_participant = compute_iid_sample_partitions(test_dataset_size, fed_train_num_participants)
    print("Dataset Distribution Computed!")
    
    # run federated training
    print("Computing Federated Training ...")
    start_training_round = 0
    for training_round in tqdm(range(start_training_round, fed_train_num_rounds)):
        print(f"\n | Federated Training Round : {training_round} | Global Model : {model_time}\n")
        
        local_weights, local_train_losses, local_val_losses = [], [], []
        idxs_participants = list(np.random.choice(range(fed_train_num_participants), fed_train_num_participants_per_round, replace=False))

        # update each local model
        print("Computing Local Updates ...")
        if fed_train_participant_order == "sequential":
            idxs_participants.sort()
            print("Local Updates of Participants will be computed in Sequential Order!")
        elif fed_train_participant_order == "random":
            print("Local Updates of Participants will be computed in Random Order!")
        else:
            raise Exception(f"Invalid Federated Training Participant Order: '{fed_train_participant_order}'! "
                            f"Only 'sequential' and 'random' are supported!")
        print(f"Participant Local Update Sequence is: {idxs_participants}")
        for idx in idxs_participants: 
            print(f"Computing Local Update of Participant {idx} ...") 
            local_sample_train_indexes = sample_train_indexes_by_participant[idx]
            local_sample_val_indexes = sample_val_indexes_by_participant[idx]
            local_sample_test_indexes = sample_test_indexes_by_participant[idx]
            local_data = SCDepthDataModule(sc_depth_hparams, 
                                           selected_train_sample_indexes=local_sample_train_indexes,
                                           selected_val_sample_indexes=local_sample_val_indexes, 
                                           selected_test_sample_indexes=local_sample_test_indexes)
            local_model = local_models[idx]
            local_trainer = Trainer(
                accelerator=device,
                max_epochs=fed_train_num_local_epochs,
                limit_train_batches=fed_train_num_local_train_batches,
                limit_val_batches=fed_train_num_local_val_batches,
                log_every_n_steps=log_every_n_steps
            )
            local_trainer.fit(local_model, local_data)
            train_epoch_losses = local_model.train_epoch_losses
            final_local_loss_train = train_epoch_losses[-1] if len(train_epoch_losses) > 0 else None
            val_epoch_losses = local_model.val_epoch_losses
            final_local_loss_val = val_epoch_losses[-1] if len(val_epoch_losses) > 0 else None

            local_train_loss = copy.deepcopy(final_local_loss_train)
            ignore = False
            if local_train_loss is None or math.isnan(local_train_loss) or math.isinf(local_train_loss):
                print(f"Local Train Loss of Participant {idx} in Round {training_round} is invalid!")
                ignore = True
            local_val_loss = copy.deepcopy(final_local_loss_val)
            if local_val_loss is None or math.isnan(local_val_loss) or math.isinf(local_val_loss):
                print(f"Local Val Loss of Participant {idx} in Round {training_round} is invalid!")
                ignore = True
            if ignore:
                print(f"Ignoring Local Update of Participant {idx} in Round {training_round}")
            else:
                local_weights.append(copy.deepcopy(local_model.state_dict()))
                local_train_losses.append(local_train_loss)
                local_val_losses.append(local_val_loss)
                print(f"Local Update of Participant {idx} Computed!")
        
        # compute performance metrics
        print("Local Updates Computed!")
        avg_local_loss_train = sum(local_train_losses) / len(local_train_losses)
        avg_local_loss_val = sum(local_val_losses) / len(local_val_losses)
        print(f"Average Local Training Loss in Round {training_round} = {avg_local_loss_train}")
        print(f"Average Local Validation Loss in Round {training_round} = {avg_local_loss_val}")

        # update global weights
        print(f"Computing Global Update ...")
        global_weights = average_weights(local_weights)
        for i in range(fed_train_num_participants):
            local_models[i] = load_weight_function(local_models[i], global_weights)
        global_model.load_state_dict(global_weights)
        print(f"Global Update Computed!")
    
    print("Federated Training Computed!")
    print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))