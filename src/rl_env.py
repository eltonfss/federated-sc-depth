from typing import Optional, Union, List

import gym
import numpy as np


class RLWeightOptimizationEnv(gym.Env):
    """
    Reinforcement Learning based Weight Optimization Environment
    """

    def __init__(
            self, w_of_w_bounds, local_model_weight_list,
            baseline_weights_of_weights, baseline_loss,
            avg_func, eval_func, random_seed
    ):
        self.local_model_weight_list = local_model_weight_list
        self.w_of_w_bounds = w_of_w_bounds
        self.avg_func = avg_func
        self.eval_func = eval_func
        self.random_seed = random_seed
        self.baseline_weights_of_weights = baseline_weights_of_weights
        self.baseline_loss = baseline_loss
        self.action_space = gym.spaces.Box(low=w_of_w_bounds[:, 0], high=w_of_w_bounds[:, 1], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(local_model_weight_list),), dtype=np.float32
        )

    def reset(self, **kwargs):
        super().reset(seed=self.random_seed, **kwargs)
        return np.array(self.baseline_weights_of_weights)  # Start with the baseline weights of weights

    def step(self, action):
        # Clip the action within the bounds
        weights_of_weights = np.clip(action, self.w_of_w_bounds[:, 0], self.w_of_w_bounds[:, 1])
        if sum(weights_of_weights) > 0:
            # Evaluate the weights of weights using the RL agent's action
            avg_weights_rl = self.avg_func(self.local_model_weight_list, weights_of_weights)
            test_loss_rl = self.eval_func(avg_weights_rl)
        else:
            test_loss_rl = 1
        reward = 1 - test_loss_rl
        # Return the new state (action) and the negative test loss as the reward
        return np.array(weights_of_weights), reward, True, {}
