import numpy as np
import scipy


def discounted_cumulative_sums(x, discount):
    return scipy.signal.lfilter(
        [1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    def __init__(
            self,
            screen_dimensions,
            available_actions_dimensions,
            size,
            gamma=0.99,
            lam=0.95):
        self.screen_buffer = np.zeros(
            (size,) + screen_dimensions, dtype=np.float32)
        self.available_actions_buffer = np.zeros(
            (size, available_actions_dimensions), dtype=np.float32)
        self.non_spatial_action_buffer = np.zeros(size, dtype=np.int32)
        self.spatial_action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros((size, 2), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(
            self,
            screen,
            available_action,
            non_spatial_action,
            spatial_action,
            reward,
            value,
            logprobability):
        self.screen_buffer[self.pointer] = screen
        self.available_actions_buffer[self.pointer] = available_action
        self.non_spatial_action_buffer[self.pointer] = non_spatial_action
        self.spatial_action_buffer[self.pointer] = spatial_action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (
            self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.screen_buffer,
            self.available_actions_buffer,
            self.non_spatial_action_buffer,
            self.spatial_action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )
