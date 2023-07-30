import gym
import numpy as np


class RLWeightOptimizationEnv(gym.Env):
    """
    Reinforcement Learning based Weight Optimization Environment
    """

    def __init__(
            self, w_of_w_bounds, local_model_weight_list,
            baseline_weights_of_weights, baseline_loss,
            norm_func, avg_func, eval_func, random_seed
    ):
        self.local_model_weight_list = local_model_weight_list
        self.w_of_w_bounds = w_of_w_bounds
        self.norm_func = norm_func
        self.avg_func = avg_func
        self.eval_func = eval_func
        self.random_seed = random_seed
        self.baseline_weights_of_weights = baseline_weights_of_weights
        self.baseline_loss = baseline_loss
        clipped_bound = 1 / (10 ** 6)  # force bounds to be greater than 0
        self.clipped_lower_bounds = np.array([max(b, clipped_bound) for b in w_of_w_bounds[:, 0]])
        self.clipped_higher_bounds = np.array([max(b, clipped_bound) for b in w_of_w_bounds[:, 1]])
        self.action_space = gym.spaces.Box(
            low=self.clipped_lower_bounds, high=self.clipped_higher_bounds, dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(local_model_weight_list),), dtype=np.float32
        )

    def reset(self, **kwargs):
        super().reset(seed=self.random_seed, **kwargs)
        return np.array(self.baseline_weights_of_weights)  # Start with the baseline weights of weights

    def step(self, action):
        # Clip the action within the bounds
        weights_of_weights = np.clip(action, self.clipped_lower_bounds, self.clipped_higher_bounds)
        if sum(weights_of_weights) > 0:
            # Evaluate the weights of weights using the RL agent's action
            normalized_w_of_w = self.norm_func(weights_of_weights)
            avg_weights_rl = self.avg_func(self.local_model_weight_list, normalized_w_of_w)
            print("Evaluating global model loss with Averaging Weights:", weights_of_weights)
            test_loss_rl = self.eval_func(avg_weights_rl)
            weights_of_weights = normalized_w_of_w
        else:
            test_loss_rl = 1
        reward = 1 - test_loss_rl
        print(f"Loss = {test_loss_rl} Reward = {reward}")
        # Return the new state (action) and the negative test loss as the reward
        return np.array(weights_of_weights), reward, False, {}
