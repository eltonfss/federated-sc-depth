import copy
import torch
import os
import json
import random
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import itertools
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


def average_weights_with_loss_optimization(
        local_model_weight_list, num_samples_for_each_local_model,
        global_model, global_data,
        global_trainer_config, grid_range_size
):

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

    # Define the range of weights to try for each local model
    weight_of_weights_range = [(i + 1) / grid_range_size for i in range(grid_range_size)]
    weight_of_weights_range.reverse()
    print("Optimizing Averaging Weights with range:", weight_of_weights_range)
    # Iterate through all possible combinations of weights for the local models
    for weights_of_weights in itertools.product(weight_of_weights_range, repeat=len(local_model_weight_list)):
        # skip combinations whose sum is greater than one
        if sum(weights_of_weights) != 1:
            continue
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
        print("Optimal averaging weights were obtained with Grid Search!")

    return best_weights, best_weights_of_weights, standard_fed_avg


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

