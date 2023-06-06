import os
import glob
import pandas as pd
import json
import numpy as np
import math
import plotly.graph_objects as go
import plotly.io as pio
from utils import compute_iid_sample_partitions
pio.kaleido.scope.mathjax = None


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


def standardize_fig(fig, x_tick_size=14, y_tick_size=20, legend_size=12, trace_size=None, show_legend=True):
    fig.update_xaxes(mirror=True,ticks='outside',showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_layout(
        plot_bgcolor='white',
        xaxis={
            'tickmode': 'array', # Set tick mode to 'array'
            'tickangle': 0, # Rotate ticks for better readability
            'mirror': True,
            'ticks': 'outside',
            'showline': True,
            'linecolor': 'black',
            'gridcolor': 'lightgrey',
            'tickfont': {'size': x_tick_size, 'family': 'Arial', 'color': 'black'} # Update font size, family, color of tick labels
        },
        yaxis={
            'mirror': True,
            'ticks': 'outside',
            'showline': True,
            'linecolor': 'black',
            'gridcolor': 'lightgrey',
            'tickfont': {'size': y_tick_size, 'family': 'Arial', 'color': 'black'} # Update font size, family, color of tick labels
        },
        title={'font': {'size': 20, 'family': 'Arial', 'color': 'black'}}, # Update font size, family, color of title
        xaxis_title={'font': {'size': 25, 'family': 'Arial', 'color': 'black'}}, # Update font size, family, color of x-axis title
        yaxis_title={'font': {'size': 25, 'family': 'Arial', 'color': 'black'}}, # Update font size, family, color of y-axis title
        legend={'font': {'size': legend_size, 'family': 'Arial', 'color': 'black'}, 'bordercolor': 'black', 'borderwidth': 0.0},
        legend_tracegroupgap=10
    )
    if trace_size:
        fig.update_traces(line={'width': trace_size}) # Update thickness
    fig.update_traces(showlegend=show_legend)
    pio.full_figure_for_development(fig, warn=False)
    return fig


def get_centralized_training_charts(centralized_training_dirpath, centralized_training_id, label=None, dataset_size_in_gb = 0, cost_multiplier = 1, model_size_mb=0, num_clients=1, show_legend=True):
    
    # Define variables
    dir_path = os.path.join(centralized_training_dirpath, centralized_training_id)

    # TODO Read CSV file
    df = pd.read_csv(os.path.join(centralized_training_dirpath, centralized_training_id, 'val_loss.csv'))
    num_steps = list(df['Step'])
    test_loss = list(df['Value'])

    # Create global test loss figures
    lowest_test_loss = [min(test_loss[:i+1]) for i in range(len(test_loss))]
    test_loss_by_training_step_fig = go.Figure()
    test_loss_by_training_step_fig.add_trace(go.Scatter(x=num_steps, y=lowest_test_loss, name=label or centralized_training_id))
    test_loss_by_training_step_fig = standardize_fig(test_loss_by_training_step_fig, show_legend=show_legend)
    
    # Create estimated communication cost figures
    communication_costs = []
    communication_cost = dataset_size_in_gb + (num_clients * model_size_mb / 1204)
    #communication_cost = dataset_size_in_gb
    for step in range(len(test_loss)):
        communication_costs.append(communication_cost)
    communication_costs = [cost * cost_multiplier for cost in communication_costs]
    communication_cost_by_training_step_fig = go.Figure()
    communication_cost_by_training_step_fig.add_trace(go.Scatter(x=num_steps, y=communication_costs, name=label or centralized_training_id))
    communication_cost_by_training_step_fig = standardize_fig(communication_cost_by_training_step_fig, show_legend=show_legend)
    
    # Create combined test_loss and communication cost figures
    test_loss_by_communication_cost_fig = go.Figure()
    test_loss_by_communication_cost_fig.add_trace(go.Scatter(x=communication_costs, y=lowest_test_loss, name=label or centralized_training_id))

    return test_loss_by_training_step_fig, communication_cost_by_training_step_fig, test_loss_by_communication_cost_fig


def get_federated_training_charts(federated_training_dirpath, round_cap, federated_training_id, label=None, sudo_centralized=False, dataset_size_in_gb = 0, cost_multiplier = 1, model_size_mb = None, show_legend=True):
    
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
    if model_size_mb is None:
        model_size_mb = sum(list(federated_training_state["global_model_bytes_by_round"].values()))/len(federated_training_state["global_model_bytes_by_round"]) / 1024 / 1024
    bytes_per_participant = model_size_mb * 2 / 1024 # Each participant uploads the entire model to the server, and downloads the updated model
    
    # compute number of steps by round (computational cost)
    num_steps_per_round = []
    total_steps = 0
    for round_num, participant_order in participant_order_by_round.items():
        #num_participants = len(participant_order)
        for participant_id in participant_order:
            num_samples_available = len(sample_train_indexes_by_participant[str(participant_id)])
            num_batches_available = num_samples_available / fed_train_local_batch_size
            num_batches_per_epoch = math.floor(min(fed_train_num_local_train_batches, num_batches_available))
            num_steps_participant = fed_train_num_local_epochs * num_batches_per_epoch
            total_steps += num_steps_participant
        num_steps_per_round.append(total_steps)

    # Extract global metrics federated_training_state
    global_test_loss = list(federated_training_state["global_test_loss_by_round"].values())
    global_test_loss = global_test_loss[:round_cap]

    # Calculate communication cost and num_steps up to the lowest loss for each round
    communication_cost = [0] * len(global_test_loss)
    num_steps = [0] * len(global_test_loss)
    lowest_loss_so_far = float('inf')
    for round_idx in range(len(global_test_loss)):
        #round_communication_cost = num_participants_per_round * bytes_per_participant * (round_idx + 1)
        round_communication_cost = 2 * num_participants * bytes_per_participant * (round_idx + 1)
        round_num_steps = num_steps_per_round[round_idx]
        if global_test_loss[round_idx] < lowest_loss_so_far:
            lowest_loss_so_far = global_test_loss[round_idx]
            #print('num_participants', num_participants)
            #print('round_communication_cost', round_communication_cost)
        else:
            round_communication_cost = communication_cost[round_idx - 1]
            round_num_steps = num_steps[round_idx - 1]
        communication_cost[round_idx] = round_communication_cost
        num_steps[round_idx] = round_num_steps
        

    # Create global test loss figures
    lowest_global_test_loss = [min(global_test_loss[:i+1]) for i in range(len(global_test_loss))]
    test_loss_by_round_fig = go.Figure()
    test_loss_by_round_fig.add_trace(go.Scatter(x=list(range(len(lowest_global_test_loss))), y=lowest_global_test_loss, name=label or federated_training_id))
    test_loss_by_training_step_fig = go.Figure()
    test_loss_by_training_step_fig.add_trace(go.Scatter(x=num_steps_per_round, y=lowest_global_test_loss, name=label or federated_training_id))
    test_loss_by_training_step_fig = standardize_fig(test_loss_by_training_step_fig, show_legend=show_legend)
    
    # Create estimated communication cost figures
    if sudo_centralized:
        communication_cost = []
        for round_num in range(len(global_test_loss)):
            communication_cost.append(dataset_size_in_gb)
        communication_cost = communication_cost[:round_cap]  
    communication_cost = [cost * cost_multiplier for cost in communication_cost]
    communication_cost_by_round_fig = go.Figure()
    communication_cost_by_round_fig.add_trace(go.Scatter(x=list(range(len(communication_cost))), y=communication_cost, name=label or federated_training_id))
    communication_cost_by_training_step_fig = go.Figure()
    communication_cost_by_training_step_fig.add_trace(go.Scatter(x=num_steps_per_round, y=communication_cost, name=label or federated_training_id))
    communication_cost_by_training_step_fig = standardize_fig(communication_cost_by_training_step_fig, show_legend=show_legend)
    
    # Create combined test_loss and communication cost figures
    test_loss_by_communication_cost_fig = go.Figure()
    test_loss_by_communication_cost_fig.add_trace(go.Scatter(x=communication_cost, y=lowest_global_test_loss, name=label or federated_training_id))
    test_loss_by_communication_cost_fig = standardize_fig(test_loss_by_communication_cost_fig, show_legend=show_legend)
    
    # Create combined test_loss and communication cost figures
    training_steps_by_round_fig = go.Figure()
    training_steps_by_round_fig.add_trace(go.Scatter(x=list(range(len(num_steps))), y=num_steps, name=label or federated_training_id))
    training_steps_by_round_fig = standardize_fig(training_steps_by_round_fig, show_legend=show_legend)
        
    return test_loss_by_round_fig, communication_cost_by_round_fig, test_loss_by_training_step_fig, communication_cost_by_training_step_fig, test_loss_by_communication_cost_fig, training_steps_by_round_fig


def get_samples_by_participant_chart(global_data: any, is_iid: bool, num_participants: int, redistribute_remaining: bool, fig: go.Figure = None, x_tick_size=25, y_tick_size=25, legend_size=25):

    if is_iid:
        train_dataset_size = global_data.get_dataset_size("train")
        sample_train_indexes_by_participant = compute_iid_sample_partitions(dataset_size=train_dataset_size, num_partitions=num_participants)
        title = "<b>Number of Samples by Participant (IID)</b>"
        name = "<b>Random Distribution (IID)</b>"
        bar_color = 'purple'
    else:
        sample_train_indexes_by_participant = compute_sample_partitions_by_drive(global_data, num_participants, redistribute_remaining)
        if redistribute_remaining:
            title = "<b>Number of Samples by Participant (By Drive With Redistribution)</b>"
            bar_color = 'rgb(255, 118, 26)'
            name = "<b>By Drive w/ Redistribution</b>"
        else:
            title = "<b>Number of Samples by Participant (By Drive)</b>"
            name = "<b>By Drive (Non-IID)</b>"
            bar_color = 'rgb(26, 118, 255)'

    sample_train_indexes_count_by_participant = {key: len(sample_train_indexes) for key, sample_train_indexes in sample_train_indexes_by_participant.items()}
    fig_x = list(sample_train_indexes_count_by_participant.keys())
    fig_y = list(sample_train_indexes_count_by_participant.values())
    if fig is None:
        fig = go.Figure(go.Bar(
            x=fig_x,
            y=fig_y,
            marker_color=bar_color,
            name=name
        ))
        fig.update_layout(
            plot_bgcolor='white',
            xaxis={
                'tickmode': 'array', # Set tick mode to 'array'
                'tickvals': fig_x, # Set tick values to unique participants
                'tickangle': 0, # Rotate ticks for better readability
                'mirror': True,
                'ticks': 'outside',
                'showline': True,
                'linecolor': 'black',
                'gridcolor': 'lightgrey',
                'tickfont': {'size': 25, 'family': 'Arial', 'color': 'black'} # Update font size, family, color of tick labels
            },
            yaxis={
                'mirror': True,
                'ticks': 'outside',
                'showline': True,
                'linecolor': 'black',
                'gridcolor': 'lightgrey',
                'tickfont': {'size': 25, 'family': 'Arial', 'color': 'black'} # Update font size, family, color of tick labels
            },
            title={'text': title, 'font': {'size': 20, 'family': 'Arial', 'color': 'black'}}, # Update font size, family, color of title
            xaxis_title={'text': "<b>Participant</b>", 'font': {'size': 30, 'family': 'Arial', 'color': 'black'}}, # Update font size, family, color of x-axis title
            yaxis_title={'text': "<b>Number of Samples</b>", 'font': {'size': 30, 'family': 'Arial', 'color': 'black'}} # Update font size, family, color of y-axis title
        )
    else:
        fig.add_trace(go.Bar(
            x=fig_x,
            y=fig_y,
            marker_color=bar_color,
            name=name
        ))

    
    #fig.update_traces(marker={'line': {'width': 2, 'color': 'black'}}) # Update thickness and color of bar outline
    pio.full_figure_for_development(fig, warn=False)
    fig = standardize_fig(fig, x_tick_size=x_tick_size, y_tick_size=y_tick_size, legend_size=legend_size)
    return fig


def compute_sample_partitions_by_drive(global_data, num_participants, redistribute_remaining):
    train_drive_ids = global_data.get_drive_ids("train")
    sample_train_indexes_by_participant = {}
    for participant_index, drive_id in enumerate(train_drive_ids):
        train_drive_samples = global_data.get_samples_by_drive_id("train", drive_id)
        train_drive_sample_indexes = [sample['sourceIndex'] for sample in train_drive_samples]
        sample_train_indexes_by_participant[str(participant_index)] = train_drive_sample_indexes
    sorted_sample_train_indexes_by_participant = sorted(sample_train_indexes_by_participant.items(),
                                                        key=lambda x: len(x[1]), reverse=True)
    for new_participant_index in range(len(sorted_sample_train_indexes_by_participant)):
        key_value_pair = sorted_sample_train_indexes_by_participant[new_participant_index]
        old_participant_index, participant_train_indexes = key_value_pair
        new_participant_index = str(new_participant_index)
        sample_train_indexes_by_participant[new_participant_index] = participant_train_indexes
    if num_participants < len(train_drive_ids):
        
        if redistribute_remaining:
            # get indexes left out
            sample_train_indexes_left_out = []
            for participant_index, sample_train_indexes in sample_train_indexes_by_participant.items():
                if int(participant_index) < num_participants:
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
            partition_size = math.ceil(number_of_samples_left_out / num_participants)
            for i in range(0, number_of_samples_left_out, partition_size):
                partitions_of_sample_train_indexes_left_out.append(
                    sample_train_indexes_left_out[i:i + partition_size]
                )
        # extend participant indexes list with partition of indexes left out
        extended_sample_train_indexes_by_participant = {}
        for participant_index, sample_train_indexes in sample_train_indexes_by_participant.items():
            if int(participant_index) >= num_participants:
                break
            extended_sample_train_indexes_by_participant[participant_index] = sample_train_indexes_by_participant[participant_index]
            if redistribute_remaining:
                participant_partition = partitions_of_sample_train_indexes_left_out[int(participant_index)]
                extended_sample_train_indexes_by_participant[participant_index].extend(participant_partition)
        sample_train_indexes_by_participant = extended_sample_train_indexes_by_participant
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
    return sample_train_indexes_by_participant
