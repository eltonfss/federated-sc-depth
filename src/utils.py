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
