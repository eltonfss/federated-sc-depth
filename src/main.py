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

from utils import set_seed, mkdir_if_missing, save_args_json, average_weights, load_weights_without_batchnorm, load_weights
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
    
    # run federated training
    print("Computing Federated Training ...")
    start_training_round = 0
    for training_round in tqdm(range(start_training_round, fed_train_num_rounds)):
        print(f"\n | Federated Training Round : {training_round} | Global Model : {model_time}\n")
        
        local_weights, local_losses = [], []
        idxs_participants = np.random.choice(range(fed_train_num_participants), fed_train_num_participants_per_round, replace=False)

        # update each local model
        print("Computing Local Updates ...")
        for idx in idxs_participants:
            print(f"Computing Local Update of Participant {idx} ...") 
            local_data = SCDepthDataModule(sc_depth_hparams) # TODO apply dataset partitioning function
            local_model = local_models[idx]
            local_trainer = Trainer(
                accelerator=device,
                max_epochs=fed_train_num_local_epochs,
                limit_train_batches=fed_train_num_local_train_batches,
                limit_val_batches=fed_train_num_local_val_batches,
                log_every_n_steps=log_every_n_steps
            )
            local_trainer.fit(local_model, local_data)
            avg_local_loss = sum(local_model.epoch_losses) / len(local_model.epoch_losses)
            local_weights.append(copy.deepcopy(local_model.state_dict()))
            local_losses.append(copy.deepcopy(avg_local_loss))
            print(f"Local Update of Participant {idx} Computed!") 
        print("Local Updates Computed!")
        avg_local_loss = sum(local_losses) / len(local_losses)
        print(f"Average Local Loss = {avg_local_loss}")

        # update global weights
        print(f"Computing Global Update ...")
        global_weights = average_weights(local_weights)
        for i in range(fed_train_num_participants):
            local_models[i] = load_weight_function(local_models[i], global_weights)
        global_model.load_state_dict(global_weights)
        print(f"Global Update Computed!")
    
    print("Federated Training Computed!")
    print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))