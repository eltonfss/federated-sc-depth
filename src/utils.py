import copy
from datetime import datetime
from sgd_regressor import OptimizedSGDRegressor

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
from rl_env import RLWeightOptimizationEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, EvalCallback

FEDERATED_TRAINING_STATE_FILENAME = "federated_training_state"
PROXY_FOR_INFINITY = 10 ** 6


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


def compute_grid_range_size(local_model_weight_list, grid_length=100):
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
        model_save_dir,
        local_model_ids,
        local_model_weight_list, num_samples_for_each_local_model,
        global_model, global_data,
        global_trainer_config, search_range_size,
        search_strategy, random_seed, aggregation_optimization_info
):
    """
    search_strategy:   'GridSearch',
                       'RandomSearch', 'ConstrainedRandomSearch',
                       'BayesianRegressionGrid', 'ConstrainedBayesianRegressionGrid',
                       'BayesianOptimization', 'ConstrainedBayesianOptimization',
                       'ReinforcementLearning', 'ConstrainedReinforcementLearning'
    """

    # validate input parameters
    valid_search_strategies = ['GridSearch',
                               'RandomSearch', 'ConstrainedRandomSearch',
                               'BayesianRegressionGrid', 'BayesianRegressionGrid',
                               'BayesianOptimization', 'ConstrainedBayesianOptimization',
                               'ReinforcementLearning', 'ConstrainedReinforcementLearning']
    assert search_strategy in valid_search_strategies, \
        f"Invalid search_strategy. Supported types: {valid_search_strategies}"
    assert len(local_model_weight_list) == len(num_samples_for_each_local_model) == len(local_model_ids), \
        'Each weight dict must have a corresponding number of samples and vice-versa!'

    # Initialize best weights and test loss with sample weighted averaging (standard FedAvg)
    best_weights, best_weights_of_weights = average_weights_by_num_samples(
        local_model_weight_list, num_samples_for_each_local_model
    )
    best_test_loss = evaluate_averaged_weights(best_weights, global_data, global_model, global_trainer_config)
    standard_fed_avg_is_best = True
    aggregation_optimization_info['standard_fed_avg_weights_of_weights'] = best_weights_of_weights
    aggregation_optimization_info['standard_fed_avg_loss'] = best_test_loss
    print("Weights of weights with standard FedAvg is:", best_weights_of_weights)
    print("Test Loss with standard FedAvg is:", best_test_loss)
    print(f"Optimizing Averaging Weights with {search_strategy} Range: {search_range_size}")

    # compute weight of weight combinations based on search strategy
    w_of_w_combinations = compute_weight_of_weight_combinations(
        best_test_loss, best_weights_of_weights, global_data, global_model, global_trainer_config,
        model_save_dir, local_model_ids, local_model_weight_list, random_seed, search_range_size, search_strategy
    )

    # Iterate through the precomputed combinations of weights for the local models
    for weights_of_weights in w_of_w_combinations:
        avg_weights = average_weights_with_weights_of_weights(local_model_weight_list, weights_of_weights)
        print("Evaluating global model loss with Averaging Weights:", weights_of_weights)
        test_epoch_loss = evaluate_averaged_weights(avg_weights, global_data, global_model, global_trainer_config)
        # Check if this combination of weights resulted in a lower test loss
        if test_epoch_loss is not None and test_epoch_loss < best_test_loss:
            best_test_loss = test_epoch_loss
            best_weights_of_weights = weights_of_weights
            best_weights = avg_weights
            standard_fed_avg_is_best = False

    # retrieve best averaging weight, loss and if its standard fed avg
    print("Best Averaging Weights is:", best_weights_of_weights)
    print("Best Loss is:", best_test_loss)
    if standard_fed_avg_is_best:
        print("Optimal averaging weights were obtained with Standard FedAvg!")
    else:
        print(f"Optimal averaging weights were obtained with {search_strategy}!")
    aggregation_optimization_info['best_weights_of_weights'] = best_weights_of_weights
    aggregation_optimization_info['best_loss'] = best_test_loss
    aggregation_optimization_info['standard_fed_avg_is_best'] = standard_fed_avg_is_best
    return best_weights, best_weights_of_weights, standard_fed_avg_is_best


def compute_weight_of_weight_combinations(best_test_loss, best_weights_of_weights, global_data, global_model,
                                          global_trainer_config, model_save_dir, local_model_ids,
                                          local_model_weight_list, random_seed,
                                          search_range_size, search_strategy):
    w_of_w_combos = []
    rng_seed = np.random.default_rng(random_seed)
    common_args = dict(
        baseline_test_loss=best_test_loss,
        baseline_weights_of_weights=best_weights_of_weights,
        model_save_dir=model_save_dir,
        local_model_ids=local_model_ids,
        local_model_weight_list=local_model_weight_list,
        random_seed=random_seed,
        search_range_size=search_range_size,
        global_data=global_data,
        global_model=global_model,
        global_trainer_config=global_trainer_config
    )
    common_args_unconstrained = dict(
        w_of_w_bounds=compute_unconstrained_w_of_w_bounds(local_model_weight_list),
        **common_args
    )
    common_args_constrained = dict(
        w_of_w_bounds=compute_constrained_w_of_w_bounds(best_weights_of_weights, local_model_weight_list),
        **common_args
    )
    if search_strategy == "BayesianRegressionGrid":
        w_of_w_combos = compute_w_of_w_combinations_with_bayesian_regression_grid(**common_args_unconstrained)
    elif search_strategy == "ConstrainedBayesianRegressionGrid":
        w_of_w_combos = compute_w_of_w_combinations_with_bayesian_regression_grid(**common_args_constrained)
    elif search_strategy == "ReinforcementLearning":
        w_of_w_combos = compute_w_of_w_combinations_with_reinforcement_learning(**common_args_unconstrained)
    elif search_strategy == "ConstrainedReinforcementLearning":
        w_of_w_combos = compute_w_of_w_combinations_with_reinforcement_learning(**common_args_constrained)
    elif search_strategy == "BayesianOptimization":
        w_of_w_combos, _, _, _ = compute_w_of_w_combinations_with_bayesian_optimization(**common_args_unconstrained)
    elif search_strategy == "ConstrainedBayesianOptimization":
        w_of_w_combos, _, _, _ = compute_w_of_w_combinations_with_bayesian_optimization(**common_args_constrained)
    elif search_strategy == "GridSearch":
        # Define grid that generates the closest number of combinations to the desired range
        w_of_w_combos = compute_w_of_w_combinations_with_grid_search(local_model_weight_list, search_range_size)
    elif search_strategy == "RandomSearch":
        # Generate random weights for each local model in the range [0, 1]
        w_of_w_combos = compute_w_of_w_combinations_with_random_search(
            local_model_weight_list, rng_seed, search_range_size,
            lambda model_weight_list, seed: [seed.random() for _ in range(len(model_weight_list))]
        )
    elif search_strategy == "ConstrainedRandomSearch":
        # Generate partially random weights for each local model in the range
        # Each weight has to be at least half of the ones with standard FedAvg and double at max
        w_of_w_combos = compute_w_of_w_combinations_with_random_search(
            local_model_weight_list, rng_seed, search_range_size,
            lambda model_weight_list, seed: [
                max(min(seed.random(), best_weights_of_weights[j] * 2), best_weights_of_weights[j] / 2)
                for j in range(len(model_weight_list))
            ]
        )
    return w_of_w_combos


def compute_w_of_w_combinations_with_bayesian_regression_grid(
        baseline_test_loss, baseline_weights_of_weights, local_model_weight_list, random_seed, search_range_size,
        global_data, global_model, global_trainer_config, w_of_w_bounds, **kwargs
):
    print("Running Bayesian Optimization")
    best_w_of_w_combos, best_losses, w_of_w_combos, losses = compute_w_of_w_combinations_with_bayesian_optimization(
        baseline_test_loss, baseline_weights_of_weights, local_model_weight_list, random_seed, search_range_size,
        global_data, global_model, global_trainer_config, w_of_w_bounds
    )

    print("Fitting Regression Model and Estimating Optimal Weights with Grid Search")
    best_w_of_w_combination, best_loss = compute_best_w_of_w_combination_with_regression_model_and_grid_search(
        global_data, global_model, global_trainer_config, local_model_weight_list, losses, w_of_w_combos
    )

    if len(best_losses) > 0 and any(best_loss >= best_loss_bayes_opt for best_loss_bayes_opt in best_losses):
        print("Test Loss of Weights found only with Bayesian Optimization are good enough!")
        w_of_w_combinations = best_w_of_w_combos
    else:
        print("Test Loss of Weights found with Bayesian Optimization + Regression Model + Grid Search are better!")
        w_of_w_combinations = [best_w_of_w_combination]

    return list(w_of_w_combinations)


def compute_best_w_of_w_combination_with_regression_model_and_grid_search(global_data, global_model,
                                                                          global_trainer_config,
                                                                          local_model_weight_list, losses,
                                                                          w_of_w_combos):
    # clean invalid losses/combos
    while PROXY_FOR_INFINITY in losses:
        invalid_loss_index = losses.index(PROXY_FOR_INFINITY)
        w_of_w_combos.pop(invalid_loss_index)
        losses.pop(invalid_loss_index)

    print("Performing Optimized Grid Search using Regression Model")
    grid_range_size = compute_grid_range_size(local_model_weight_list)
    grid_w_of_w_combos = compute_w_of_w_combinations_with_grid_search(local_model_weight_list, grid_range_size)
    regression_model = OptimizedSGDRegressor()
    regression_model.fit(np.array(w_of_w_combos), np.array(losses))
    grid_test_losses = list(regression_model.predict(np.array(grid_w_of_w_combos)))
    best_grid_test_loss = min(grid_test_losses)
    print("Best Regression Model: ", regression_model.get_best_configuration())
    best_w_of_w = normalize_weights_of_weights(grid_w_of_w_combos[grid_test_losses.index(best_grid_test_loss)])
    print("Best Averaging Weights Found By Regression Model: ", best_w_of_w)
    print("Estimated Test Loss:", best_grid_test_loss)
    print("Computing Actual Test Loss...")
    avg_weights = average_weights_with_weights_of_weights(local_model_weight_list, best_w_of_w)
    test_loss = evaluate_averaged_weights(avg_weights, global_data, global_model, global_trainer_config)
    return best_w_of_w, test_loss


def compute_constrained_w_of_w_bounds(best_weights_of_weights, local_model_weight_list,
                                      freedom_ratio=PROXY_FOR_INFINITY):
    # Set bounds for the weights of weights (values between half of the ones with standard FedAvg and double)
    w_of_w_bounds = np.array([
        (
            best_weights_of_weights[j] / freedom_ratio,
            min(best_weights_of_weights[j] * freedom_ratio, 1)  # maximum of 1
        ) for j in
        range(len(local_model_weight_list))
    ])
    return w_of_w_bounds


def compute_unconstrained_w_of_w_bounds(local_model_weight_list):
    # Set bounds for the weights of weights (values between 0 and 1)
    w_of_w_bounds = np.array([(0.0, 1.0) for _ in range(len(local_model_weight_list))])
    return w_of_w_bounds


def compute_w_of_w_combinations_with_reinforcement_learning(
        baseline_test_loss, baseline_weights_of_weights,
        model_save_dir, local_model_ids,
        local_model_weight_list, random_seed, search_range_size,
        global_data, global_model, global_trainer_config, w_of_w_bounds
):

    # sort everything by id order to make reuse feasible
    local_model_weight_by_id = {
        str(local_model_ids[i]): local_model_weight_list[i] for i in range(len(local_model_ids))
    }
    sorted_local_model_ids = sorted([str(model_id) for model_id in local_model_ids])
    sorted_local_model_weight_list = [local_model_weight_by_id[model_id] for model_id in sorted_local_model_ids]
    baseline_weights_of_weights_by_model_id = {
        str(local_model_ids[i]): baseline_weights_of_weights[i] for i in range(len(local_model_ids))
    }
    sorted_baseline_weights_of_weights = [
        baseline_weights_of_weights_by_model_id[model_id] for model_id in sorted_local_model_ids
    ]

    # Define the RL environment for the optimization problem
    rl_env = RLWeightOptimizationEnv(
        w_of_w_bounds=w_of_w_bounds, local_model_weight_list=sorted_local_model_weight_list, random_seed=random_seed,
        baseline_weights_of_weights=sorted_baseline_weights_of_weights, baseline_loss=baseline_test_loss,
        norm_func=normalize_weights_of_weights,
        avg_func=lambda w, w_of_w: average_weights_with_weights_of_weights(w, w_of_w),
        eval_func=lambda avg_w: evaluate_averaged_weights(avg_w, global_data, global_model, global_trainer_config)
    )

    rl_model_id = f"""rl-model-participants-{"-".join(sorted_local_model_ids)}"""
    rl_model_dirpath = os.path.join(model_save_dir, 'rl-models')
    rl_model_filepath = os.path.join(rl_model_dirpath, f"{rl_model_id}.zip")
    if os.path.exists(rl_model_filepath):
        print(f"Loading preexisting PPO model from {rl_model_filepath}")
        model = PPO.load(rl_model_filepath, env=rl_env)
    else:
        # Define the RL agent policy and the policy network architecture
        max_number_of_neurons = int(2**(len(local_model_weight_list)))
        net_arch = [max_number_of_neurons, int(max_number_of_neurons/2)]
        policy_kwargs = dict(net_arch=net_arch)
        print(f"Creating new PPO model with n_steps={search_range_size} and net_arch={net_arch}")
        model = PPO("MlpPolicy", rl_env,
                    verbose=1,
                    policy_kwargs=policy_kwargs,
                    n_steps=max(search_range_size, 2),
                    tensorboard_log=rl_model_dirpath)

    # Define the early stopping callback by Number of steps without improvement before stopping
    early_stopping_patience = max(10, len(local_model_ids) * 10)
    early_stopping = StopTrainingOnNoModelImprovement(early_stopping_patience, verbose=1)

    # Define the evaluation callback.
    eval_callback = EvalCallback(rl_env,
                                 best_model_save_path=f"best-{rl_model_filepath}",
                                 callback_after_eval=early_stopping,
                                 eval_freq=int(search_range_size/10),
                                 n_eval_episodes=int(search_range_size/10),
                                 verbose=1)

    # Train the RL agent
    print(f"Training PPO model with total_timesteps={search_range_size}")
    model.learn(total_timesteps=search_range_size, progress_bar=True, callback=[eval_callback, early_stopping])
    # Get the best action (weights of weights) from the learned policy
    best_action, _ = model.predict(np.array(sorted_baseline_weights_of_weights))
    sorted_best_weight_of_weights = normalize_weights_of_weights(best_action.tolist())
    best_w_of_w_by_model_id = {
        str(sorted_local_model_ids[i]): sorted_best_weight_of_weights[i] for i in range(len(sorted_local_model_ids))
    }
    best_weight_of_weights = [best_w_of_w_by_model_id[str(model_id)] for model_id in local_model_ids]
    print(f"Best weight of weights predicted by PPO Model: {best_weight_of_weights}")

    # save RL agent state
    print(f"Saving PPO model at {rl_model_filepath}")
    if not os.path.exists(rl_model_dirpath):
        os.makedirs(rl_model_dirpath)
    model.save(rl_model_filepath)

    return [best_weight_of_weights]


def compute_w_of_w_combinations_with_random_search(
        local_model_weight_list, rng_seed, search_range_size,
        weights_of_weights_sampling_function, **kwargs
):
    w_of_w_combinations = []
    for i in range(search_range_size):
        # Generate random weights for each local model in the range [0, 1]
        weights_of_weights = []
        while sum(weights_of_weights) == 0:
            weights_of_weights = weights_of_weights_sampling_function(local_model_weight_list, rng_seed)
        w_of_w_combinations.append(normalize_weights_of_weights(weights_of_weights))
    return w_of_w_combinations


def compute_w_of_w_combinations_with_grid_search(local_model_weight_list, search_range_size, **kwargs):
    # Define the range of weights to try for each local model
    grid_length = compute_optimal_grid_length(search_range_size, local_model_weight_list)
    weight_of_weights_range = [i / grid_length for i in range(grid_length + 1)]
    weight_of_weights_range.reverse()
    w_of_w_combinations = list(itertools.product(weight_of_weights_range, repeat=len(local_model_weight_list)))
    weights_of_weights_list_pruned = []
    for weights_of_weights in w_of_w_combinations:
        if sum(weights_of_weights) > 0:
            weights_of_weights_list_pruned.append(normalize_weights_of_weights(weights_of_weights))
    w_of_w_combinations = weights_of_weights_list_pruned
    return w_of_w_combinations


def compute_w_of_w_combinations_with_bayesian_optimization(
        baseline_test_loss, baseline_weights_of_weights, local_model_weight_list, random_seed, search_range_size,
        global_data, global_model, global_trainer_config, w_of_w_bounds, **kwargs
):
    best_w_of_w_combinations = []
    best_test_losses = []

    # Perform Bayesian optimization using gp_minimize
    def evaluate_weights_of_weights_for_bayesian_optimization(w_of_w):
        try:
            w_of_w = normalize_weights_of_weights(w_of_w)
            avg_weights_bo = average_weights_with_weights_of_weights(local_model_weight_list, w_of_w)
            test_loss = evaluate_averaged_weights(avg_weights_bo, global_data, global_model, global_trainer_config)
        except:
            test_loss = PROXY_FOR_INFINITY
        return test_loss if test_loss is not None else PROXY_FOR_INFINITY

    n_initial_random_points = int(len(w_of_w_bounds) * search_range_size)
    n_optimization_iterations = max(n_initial_random_points*2, 5)
    result = gp_minimize(
        evaluate_weights_of_weights_for_bayesian_optimization,
        dimensions=w_of_w_bounds,
        n_calls=n_optimization_iterations,
        x0=[baseline_weights_of_weights],  # initialize with standardFedAvg weights
        y0=[baseline_test_loss],  # initialize with standardFedAvg loss
        n_initial_points=n_initial_random_points,
        verbose=True,
        random_state=random_seed,
        n_jobs=-1
    )
    weights_of_weights_samples = result.x_iters
    test_loss_samples = list(result.func_vals)
    # Extract the best weights of weights from the result in order to compare with standard fed avg
    weights_of_weights = normalize_weights_of_weights(result.x)
    if baseline_weights_of_weights != weights_of_weights:
        best_w_of_w_combinations = [weights_of_weights]
        best_test_losses = [result.fun]
    return best_w_of_w_combinations, best_test_losses, weights_of_weights_samples, test_loss_samples


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

