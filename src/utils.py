import copy
from datetime import datetime

import torch
import os
import json
import random
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import itertools
from skopt import gp_minimize
from pytorch_lightning import Trainer

FEDERATED_TRAINING_STATE_FILENAME = "federated_training_state"


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


def backup_and_restore_federated_training_state(save_directory_path, federated_training_state):
    backup_federated_training_state(save_directory_path, federated_training_state)
    return restore_federated_training_state(save_directory_path)


def backup_federated_training_state(save_directory_path, federated_training_state):
    save_dict_or_list_as_json(save_directory_path, FEDERATED_TRAINING_STATE_FILENAME, federated_training_state)
    snapshots_path = os.path.join(save_directory_path, "snapshots")
    previous_snapshots_filenames = os.listdir(snapshots_path) if os.path.exists(snapshots_path) else []
    previous_snapshot_filename = None
    if len(previous_snapshots_filenames) >= 3:
        previous_snapshots_filenames.sort()
        previous_snapshot_filename = previous_snapshots_filenames[0]
    snapshot_filename = f"{FEDERATED_TRAINING_STATE_FILENAME}_{datetime.now().strftime('%d_%m_%Y_%H:%M:%S')}"
    save_dict_or_list_as_json(snapshots_path, snapshot_filename, federated_training_state)
    if previous_snapshot_filename:
        os.remove(os.path.join(snapshots_path, previous_snapshot_filename))


def restore_federated_training_state(save_directory_path):
    return read_dict_or_list_from_json(save_directory_path, FEDERATED_TRAINING_STATE_FILENAME)


def save_dict_or_list_as_json(save_directory_path, filename_without_extension, dict_or_list,
                              indent=None, sort_keys=False):
    mkdir_if_missing(save_directory_path)
    filepath = os.path.join(save_directory_path, f'{filename_without_extension}.json')
    with open(filepath, "w") as f:
        json.dump(dict_or_list, f, indent=indent, sort_keys=sort_keys)


def read_dict_or_list_from_json(save_directory_path, filename_without_extension):
    filename = os.path.join(save_directory_path, f"{filename_without_extension}.json")
    with open(filename, "r") as f:
        return json.load(f)


def average_weights(w, avg_weights=None):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w[0].keys():
        for i in range(1, len(w)):
            w_avg[key] = w_avg[key] + w[i][key]

        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_weights_by_num_samples(w, num_samples):
    """
    Returns the weighted average of the weights based on the number of samples used by each client.
    """

    assert len(w) == len(num_samples), 'Each weight dict must have a corresponding number of samples and vice-versa!'

    total_samples = sum(num_samples)
    weight = w[0]
    sample_weight = (num_samples[0] / total_samples)
    sample_weights = [sample_weight]
    avg_weights = copy.deepcopy(weight)
    for key in weight.keys():
        sample_weighted_weights = weight[key] * sample_weight
        if avg_weights[key].dtype == torch.int64:
            sample_weighted_weights = sample_weighted_weights.long()
        avg_weights[key] = sample_weighted_weights

    for i in range(1, len(w)):
        weight = w[i]
        sample_weight = (num_samples[i] / total_samples)
        sample_weights.append(sample_weight)
        for key in weight.keys():
            sample_weighted_weights = weight[key] * sample_weight
            if avg_weights[key].dtype == torch.int64:
                sample_weighted_weights = sample_weighted_weights.long()
            avg_weights[key] += sample_weighted_weights

    return avg_weights, sample_weights


def compute_grid_range_size(local_model_weight_list, grid_length):
    if grid_length <= 0:
        return 0
    # Generate the list of values for y
    weights_of_weights = [i / grid_length for i in range(grid_length + 1)]
    # Count the combinations where the sum of elements is 1
    weights_of_weights_grid = itertools.product(weights_of_weights, repeat=len(local_model_weight_list))
    grid_range_size = 0
    for w_of_w in weights_of_weights_grid:
        if sum(w_of_w) > 0 and sum(normalize_weights_of_weights(w_of_w)) == 1:
            grid_range_size += 1
    return grid_range_size


def compute_optimal_grid_length(desired_grid_range_size, local_model_weight_list, lower_bound=1, upper_bound=100):
    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        grid_range_size = compute_grid_range_size(local_model_weight_list, mid)
        if grid_range_size == desired_grid_range_size:
            return mid
        elif grid_range_size < desired_grid_range_size:
            lower_bound = mid + 1
        else:
            upper_bound = mid - 1
    # If an exact match is not found, return the closest value to desired_grid_range_size
    diff_lower_bound = compute_grid_range_size(local_model_weight_list, lower_bound) - desired_grid_range_size
    diff_upper_bound = desired_grid_range_size - compute_grid_range_size(local_model_weight_list, upper_bound)
    result = lower_bound if diff_lower_bound < diff_upper_bound else upper_bound
    return result if result > 0 else 1


def average_weights_optimization_by_search(
        local_model_weight_list, num_samples_for_each_local_model,
        global_model, global_data,
        global_trainer_config, search_range_size,
        search_strategy, random_seed
):
    """
    search_strategy: 'GridSearch', 'RandomSearch', 'BayesianOptimization',
    'ConstrainedRandomSearch', 'ConstrainedBayesianOptimization', 'ReinforcementLearning'
    """
    valid_search_strategies = ['GridSearch', 'RandomSearch', 'BayesianOptimization',
                               'ConstrainedRandomSearch', 'ConstrainedBayesianOptimization', 'ReinforcementLearning']
    assert search_strategy in valid_search_strategies, \
        f"Invalid search_strategy. Supported types: {valid_search_strategies}"
    assert len(local_model_weight_list) == len(num_samples_for_each_local_model), \
        'Each weight dict must have a corresponding number of samples and vice-versa!'

    # Initialize best weights and test loss with sample weighted averaging (standard FedAvg)
    best_weights, best_weights_of_weights = average_weights_by_num_samples(
        local_model_weight_list, num_samples_for_each_local_model
    )
    best_test_loss = evaluate_averaged_weights(best_weights, global_data, global_model, global_trainer_config)
    standard_fed_avg = True
    print("Weights of weights with standard FedAvg is:", best_weights_of_weights)
    print("Test Loss with standard FedAvg is:", best_test_loss)
    print(f"Optimizing Averaging Weights with {search_strategy} Range: {search_range_size}")

    # compute weight combinations based on search strategy
    weights_of_weights_list = []
    rng_seed = np.random.default_rng(random_seed)
    if search_strategy == "BayesianOptimization":
        # Set bounds for the weights of weights (values between 0 and 1)
        w_of_w_bounds = [(0.0, 1.0) for _ in range(len(local_model_weight_list))]
        weights_of_weights_list = compute_weights_of_weights_list_with_bayesian_optimization(
            best_test_loss, best_weights_of_weights, local_model_weight_list, random_seed, search_range_size,
            global_data, global_model, global_trainer_config, w_of_w_bounds
        )
    elif search_strategy == "ConstrainedBayesianOptimization":
        # Set bounds for the weights of weights (values between half of the ones with standard FedAvg and double)
        w_of_w_bounds = [
            (best_weights_of_weights[j]/2, best_weights_of_weights[j]*2) for j in range(len(local_model_weight_list))
        ]
        weights_of_weights_list = compute_weights_of_weights_list_with_bayesian_optimization(
            best_test_loss, best_weights_of_weights, local_model_weight_list, random_seed, search_range_size,
            global_data, global_model, global_trainer_config, w_of_w_bounds
        )
    elif search_strategy == "GridSearch":
        # Define grid that generates the closest number of combinations to the desired range
        weights_of_weights_list = compute_weights_of_weights_list_with_grid_search(
            local_model_weight_list, search_range_size
        )
    elif search_strategy == "RandomSearch":
        # Generate random weights for each local model in the range [0, 1]
        weights_of_weights_list = compute_weights_of_weights_list_with_random_search(
            local_model_weight_list, rng_seed, search_range_size,
            lambda model_weight_list, seed: [seed.random() for _ in range(len(model_weight_list))]
        )
    elif search_strategy == "ConstrainedRandomSearch":
        # Generate partially random weights for each local model in the range
        # Each weight has to be at least half of the ones with standard FedAvg and double at max
        weights_of_weights_list = compute_weights_of_weights_list_with_random_search(
            local_model_weight_list, rng_seed, search_range_size,
            lambda model_weight_list, seed: [
                max(min(seed.random(), best_weights_of_weights[j] * 2), best_weights_of_weights[j] / 2)
                for j in range(len(model_weight_list))
            ]
        )

    # Iterate through the precomputed combinations of weights for the local models
    for weights_of_weights in weights_of_weights_list:
        avg_weights = average_weights_with_weights_of_weights(local_model_weight_list, weights_of_weights)
        print("Evaluating global model loss with Averaging Weights:", weights_of_weights)
        test_epoch_loss = evaluate_averaged_weights(avg_weights, global_data, global_model, global_trainer_config)
        # Check if this combination of weights resulted in a lower test loss
        if test_epoch_loss is not None and test_epoch_loss < best_test_loss:
            best_test_loss = test_epoch_loss
            best_weights_of_weights = weights_of_weights
            best_weights = avg_weights
            standard_fed_avg = False
    print("Best Averaging Weights is:", best_weights_of_weights)
    print("Best Loss is:", best_test_loss)

    if standard_fed_avg:
        print("Optimal averaging weights were obtained with Standard FedAvg!")
    else:
        print(f"Optimal averaging weights were obtained with {search_strategy}!")

    return best_weights, best_weights_of_weights, standard_fed_avg


def compute_weights_of_weights_list_with_random_search(
        local_model_weight_list, rng_seed, search_range_size,
        weights_of_weights_sampling_function
):
    weights_of_weights_list = []
    for i in range(search_range_size):
        # Generate random weights for each local model in the range [0, 1]
        weights_of_weights = []
        while sum(weights_of_weights) == 0:
            weights_of_weights = weights_of_weights_sampling_function(local_model_weight_list, rng_seed)
        weights_of_weights_list.append(normalize_weights_of_weights(weights_of_weights))
    return weights_of_weights_list


def compute_weights_of_weights_list_with_grid_search(local_model_weight_list, search_range_size):
    # Define the range of weights to try for each local model
    grid_length = compute_optimal_grid_length(search_range_size, local_model_weight_list)
    weight_of_weights_range = [i / grid_length for i in range(grid_length + 1)]
    weight_of_weights_range.reverse()
    weights_of_weights_list = list(itertools.product(weight_of_weights_range, repeat=len(local_model_weight_list)))
    weights_of_weights_list_pruned = []
    for weights_of_weights in weights_of_weights_list:
        if sum(weights_of_weights) > 0:
            weights_of_weights_list_pruned.append(normalize_weights_of_weights(weights_of_weights))
    weights_of_weights_list = weights_of_weights_list_pruned
    return weights_of_weights_list


def compute_weights_of_weights_list_with_bayesian_optimization(
        best_test_loss, best_weights_of_weights, local_model_weight_list, random_seed, search_range_size,
        global_data, global_model, global_trainer_config, w_of_w_bounds
):
    weights_of_weights_list = []

    # Perform Bayesian optimization using gp_minimize
    def evaluate_weights_of_weights_for_bayesian_optimization(w_of_w):
        proxy_for_infinite_loss = 100000
        try:
            w_of_w = normalize_weights_of_weights(w_of_w)
            avg_weights_bo = average_weights_with_weights_of_weights(local_model_weight_list, w_of_w)
            test_loss = evaluate_averaged_weights(avg_weights_bo, global_data, global_model, global_trainer_config)
        except:
            test_loss = proxy_for_infinite_loss
        return test_loss if test_loss is not None else proxy_for_infinite_loss

    result = gp_minimize(
        evaluate_weights_of_weights_for_bayesian_optimization,
        dimensions=w_of_w_bounds,
        n_calls=search_range_size if search_range_size >= 5 else 5,
        x0=[best_weights_of_weights],  # initialize with standardFedAvg weights
        y0=[best_test_loss],  # initialize with standardFedAvg loss
        n_initial_points=5,
        verbose=True,
        random_state=random_seed,
        n_jobs=1
    )
    # Extract the best weights of weights from the result in order to compare with standard fed avg
    weights_of_weights = normalize_weights_of_weights(result.x)
    if best_weights_of_weights != weights_of_weights:
        weights_of_weights_list = [weights_of_weights]
    return weights_of_weights_list


def average_weights_with_weights_of_weights(local_model_weight_list, weights_of_weights):
    # Compute the weighted average of the local weights using the current combination
    avg_weights = {}
    for i in range(len(local_model_weight_list)):
        weight = local_model_weight_list[i]
        if i == 0:
            avg_weights = copy.deepcopy(weight)
        for key in weight.keys():
            weighted_weights = weight[key] * weights_of_weights[i]
            if avg_weights[key].dtype == torch.int64:
                weighted_weights = weighted_weights.long()
            if i == 0:
                avg_weights[key] = weighted_weights
            else:
                avg_weights[key] += weighted_weights
    return avg_weights


def normalize_weights_of_weights(weights_of_weights):
    # Normalize the weights to have a sum of 1
    weights_of_weights_sum = sum(weights_of_weights)
    weights_of_weights = [w_of_w / weights_of_weights_sum for w_of_w in weights_of_weights]
    return weights_of_weights


def evaluate_averaged_weights(avg_weights, global_data, global_model, global_trainer_config):
    # Update the global model's weights with the averaged weights
    global_model.load_state_dict(avg_weights)
    # Test the global model and compute the test loss
    global_trainer = Trainer(**global_trainer_config)
    test_config = dict(model=global_model, datamodule=global_data)
    global_trainer.test(**test_config)
    test_epoch_losses = global_model.test_epoch_losses
    test_epoch_loss = test_epoch_losses[-1] if len(test_epoch_losses) > 0 else None
    return test_epoch_loss


def average_weights_weighting_by_loss(weights, losses):
    """
    Returns the weighted average of the weights, where the weight of each
    local model is proportional to its performance on the validation set.
    """
    w_avg = copy.deepcopy(weights[0])
    for key in weights[0].keys():
        w_avg[key] = torch.zeros_like(weights[0][key]).float()  # set data type to Float
        for i in range(len(weights)):
            weight_i = 1 / losses[i]
            w_avg[key] += weights[i][key] * weight_i
    return w_avg


def load_weights_without_batchnorm(model, w):
    """
    Returns the average of the weights.
    """
    model.load_state_dict(
        {k: v for k, v in w.items() if "bn" not in k and "running" not in k},
        strict=False,
    )
    return model


def load_weights(model, w):
    """
    Returns the average of the weights.
    """
    model.load_state_dict({k: v for k, v in w.items()}, strict=False)
    return model

def visualize_image(image):
    """
    tensor image: (3, H, W)
    """
    x = (image.cpu() * 0.225 + 0.45)
    return x


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8)  # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_


def compute_iid_sample_partitions(dataset_size, num_partitions):
    """
    Partition datasets sample indexes as I.I.D.
    :param dataset_size:
    :param num_partitions:
    :return: sample_indexes_by_partition
    """
    num_samples = int(dataset_size / num_partitions)
    sample_indexes_by_partition, available_sample_indexes = {}, [sample_index for sample_index in range(dataset_size)]
    for partition_index in range(num_partitions):
        sample_indexes_by_partition[str(partition_index)] = set(np.random.choice(available_sample_indexes, num_samples,
                                                                            replace=False))
        sample_indexes_by_partition[str(partition_index)] = [int(i) for i in sample_indexes_by_partition[str(partition_index)]]
        available_sample_indexes = list(set(available_sample_indexes) - set(sample_indexes_by_partition[str(partition_index)]))
    return sample_indexes_by_partition


def estimate_model_size(weights):
    """
    Estimates the size of the model in bytes based on the weights.
    """
    # Concatenate the tensors into a single vector
    vector = torch.nn.utils.parameters_to_vector(list(weights.values()))
    # Compute the size of the vector
    size_bytes = vector.numel() * vector.element_size()
    return size_bytes

