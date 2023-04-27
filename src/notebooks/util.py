import os
import glob
import pandas as pd
import json
import numpy as np
import math
import plotly.graph_objects as go


def get_dir_size(dir_path, formats=None):
    total_size = 0
    filtered_size = 0
    total_files = 0
    filtered_files = 0

    for root, dirs, files in os.walk(dir_path):
        for f in files:
            total_files += 1
            file_path = os.path.join(root, f)
            total_size += os.path.getsize(file_path)
            if formats and f.split('.')[-1] in formats:
                filtered_files += 1
                filtered_size += os.path.getsize(file_path)

    total_size_gb = round(total_size / (1024**3), 2)
    filtered_size_gb = round(filtered_size / (1024**3), 2)

    return total_files, total_size_gb, filtered_files, filtered_size_gb

def get_federated_training_charts(federated_training_dirpath, round_cap, federated_training_id, label=None, sudo_centralized=False, dataset_size_in_gb = 0, cost_multiplier = 1):
    
    # Define variables
    dir_path = os.path.join(federated_training_dirpath, federated_training_id)
    assert os.path.exists(dir_path), 'federated_training_dirpath does not exist!'

    # Read JSON file
    with open(os.path.join(dir_path, 'federated_training_state.json'), 'r') as f:
        federated_training_state = json.load(f)
        
    config_args = federated_training_state['config_args']
    num_participants = config_args['fed_train_num_participants']
    frac_participants_per_round = config_args['fed_train_frac_participants_per_round']
    num_rounds = config_args['fed_train_num_rounds']
    fed_train_num_local_train_batches = config_args['fed_train_num_local_train_batches']
    fed_train_local_batch_size = config_args['fed_train_local_batch_size']
    fed_train_num_local_epochs = config_args['fed_train_num_local_epochs']
    fed_train_num_participants = config_args['fed_train_num_participants']
    fed_train_frac_participants_per_round = config_args['fed_train_frac_participants_per_round']
    sample_train_indexes_by_participant = federated_training_state['sample_train_indexes_by_participant']
    participant_order_by_round = federated_training_state['participant_order_by_round']
    
    num_participants_per_round = num_participants * frac_participants_per_round
    model_size_mb = 111.417
    bytes_per_participant = model_size_mb * 2 / 1024 # Each participant uploads the entire model to the server, and downloads the updated model
    
    # compute number of steps by round (computational cost)
    num_steps_per_round = []
    total_steps = 0
    for round_num, participant_order in participant_order_by_round.items():
        num_participants = len(participant_order)
        for participant_id in participant_order:
            num_samples_available = len(sample_train_indexes_by_participant[str(participant_id)])
            num_batches_available = num_samples_available / fed_train_local_batch_size
            num_batches_per_epoch = math.floor(min(fed_train_num_local_train_batches, num_batches_available))
            num_steps_participant = fed_train_num_local_epochs * num_batches_per_epoch * fed_train_local_batch_size
            total_steps += num_steps_participant
        num_steps_per_round.append(total_steps)

    # Extract global metrics federated_training_state
    global_test_loss = list(federated_training_state["global_test_loss_by_round"].values())
    global_test_loss = global_test_loss[:round_cap]

    # Calculate communication cost up to the lowest loss for each round
    communication_cost = [0] * len(global_test_loss)
    lowest_loss_so_far = float('inf')
    for round_idx in range(len(global_test_loss)):
        round_communication_cost = num_participants_per_round * bytes_per_participant * (round_idx + 1)
        if global_test_loss[round_idx] < lowest_loss_so_far:
            lowest_loss_so_far = global_test_loss[round_idx]
        else:
            round_communication_cost = communication_cost[round_idx - 1]
        communication_cost[round_idx] = round_communication_cost

    # Create global test loss figures
    lowest_global_test_loss = [min(global_test_loss[:i+1]) for i in range(len(global_test_loss))]
    test_loss_by_round_fig = go.Figure()
    test_loss_by_round_fig.add_trace(go.Scatter(x=list(range(len(lowest_global_test_loss))), y=lowest_global_test_loss, name=label or federated_training_id))
    test_loss_by_training_step_fig = go.Figure()
    test_loss_by_training_step_fig.add_trace(go.Scatter(x=num_steps_per_round, y=lowest_global_test_loss, name=label or federated_training_id))
    
    # Create estimated communication cost figures
    if sudo_centralized:
        communication_cost = [dataset_size_in_gb]
        for round_num in range(1, len(global_test_loss)):
            communication_cost.append(0)
        communication_cost = communication_cost[:round_cap]  
    communication_cost = [cost * cost_multiplier for cost in communication_cost]
    communication_cost_by_round_fig = go.Figure()
    communication_cost_by_round_fig.add_trace(go.Scatter(x=list(range(len(communication_cost))), y=communication_cost, name=label or federated_training_id))
    communication_cost_by_training_step_fig = go.Figure()
    communication_cost_by_training_step_fig.add_trace(go.Scatter(x=num_steps_per_round, y=communication_cost, name=label or federated_training_id))
    
    # Create combined test_loss and communication cost figures
    test_loss_by_communication_cost_fig = go.Figure()
    test_loss_by_communication_cost_fig.add_trace(go.Scatter(x=communication_cost, y=lowest_global_test_loss, name=label or federated_training_id))

        
    return test_loss_by_round_fig, communication_cost_by_round_fig, test_loss_by_training_step_fig, communication_cost_by_training_step_fig, test_loss_by_communication_cost_fig