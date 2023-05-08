import copy
import torch
import numpy as np
import os
import json
import random
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

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


def average_weights_by_num_samples(w, num_samples, avg_weights=None):
    """
    Returns the weighted average of the weights based on the number of samples used by each client.
    """
    if avg_weights is None:
        avg_weights = copy.deepcopy(w[0])
    else:
        for key in avg_weights.keys():
            avg_weights[key].zero_()

    assert len(w) == len(num_samples), 'Each weight dict must have a corresponding number of samples and vice-versa!'

    total_samples = sum(num_samples)
    for i in range(len(w)):
        weight = w[i]
        sample_weight = (num_samples[i] / total_samples)
        for key in weight.keys():
            sample_weighted_weights = weight[key] * sample_weight
            if avg_weights[key].dtype == torch.int64:
                sample_weighted_weights = sample_weighted_weights.long()
            avg_weights[key] += sample_weighted_weights

    return avg_weights


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
