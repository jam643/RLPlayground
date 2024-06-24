import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass
from matplotlib import pyplot as plt
import skimage.measure
import time
from typing import Optional
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
from scipy import integrate
from enum import Enum
from abc import ABC, abstractmethod

from dubins_py import get_multi_waypoint_from_np, is_path_valid

from stable_baselines3.common.env_checker import check_env


@dataclass
class State:
    path_index: int
    lateral_dev: float
    local_heading: float
    steering_curvature: float


@dataclass
class Action:
    steering_curvature_change: float


def car_polygon(x, y, heading, length=5.0, width=2.0):
    """
    Computes the car's polygon at a given pose.

    Parameters:
    x (float): The x position of the car's center.
    y (float): The y position of the car's center.
    heading (float): The heading angle of the car in radians.
    length (float): The length of the car (default is 2.0).
    width (float): The width of the car (default is 1.0).

    Returns:
    tuple: Two numpy arrays representing the x and y coordinates of the car's corners.
    """
    # Define the corner points relative to the car's center
    corners = np.array([
        [-length / 2, -width / 2],
        [length / 2, -width / 2],
        [length / 2, width / 2],
        [-length / 2, width / 2]
    ])

    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading), np.cos(heading)]
    ])

    # Rotate and translate the corner points
    rotated_corners = corners @ rotation_matrix.T
    translated_corners = rotated_corners + np.array([x, y])

    # Separate the x and y coordinates
    x_coords = translated_corners[:, 0]
    y_coords = translated_corners[:, 1]

    return x_coords, y_coords


def motion_model(
        state: State, action: Action, max_steering_curvature: float, path: np.ndarray, ds: float
) -> State:
    steering_curvature = state.steering_curvature + action.steering_curvature_change * ds
    steering_curvature = np.clip(steering_curvature, -max_steering_curvature, max_steering_curvature)
    path_curvature = path[state.path_index, 3]
    path_length_per_ds = (1 - state.lateral_dev * path_curvature) / np.cos(state.local_heading)
    local_heading = state.local_heading + steering_curvature * ds * path_length_per_ds - path_curvature * ds
    lateral_dev = state.lateral_dev + np.sin(local_heading) * path_length_per_ds
    path_index = state.path_index + 1

    return State(path_index=path_index, lateral_dev=lateral_dev, local_heading=local_heading,
                 steering_curvature=steering_curvature)


def mean_pooling_1d(arr, pool_size):
    # Ensure the length of arr is a multiple of pool_size
    assert len(arr) % pool_size == 0, "Array length must be a multiple of the pooling size"

    # Reshape the array to a 2D array where each row is a pool
    reshaped_arr = arr.reshape(-1, pool_size)

    # Calculate the mean of each pool
    pooled_arr = reshaped_arr.mean(axis=1)

    return pooled_arr


@dataclass
class Observation:
    lateral_dev: float
    local_heading: float
    steering_curvature: float
    curvature_lookahead: np.ndarray

    @staticmethod
    def build(state: State, path: np.ndarray, max_lateral_dev: float, N_lookahead: int, N_lookahead_scale: int):
        lookahead = path[state.path_index:state.path_index + N_lookahead * N_lookahead_scale, 3]
        lookahead = mean_pooling_1d(lookahead, N_lookahead_scale)

        # build curvature lookahaed
        return Observation(
            lateral_dev=state.lateral_dev,
            local_heading=state.local_heading,
            steering_curvature=state.steering_curvature,
            curvature_lookahead=lookahead
        )

    def to_np(self):
        state = np.array(
            [
                self.lateral_dev / 5.0,
                self.local_heading,
                self.steering_curvature
            ]
        )
        full_obs = (np.concatenate([state, self.curvature_lookahead]).astype(np.float32))
        return full_obs


class Reward:
    @dataclass
    class Params:
        lateral_acc_weight: float = 200.0
        curvature_change_weight: float = 20
        lateral_dev_weight: float = 0
        survival_reward = 1
        violation_buffer = 1

    def __init__(self, params: Params):
        self.params = params

    def compute_reward(self, obs: Observation, action: Action):
        curvature_cost = self.params.lateral_acc_weight * obs.steering_curvature ** 2
        curvature_change_cost = self.params.curvature_change_weight * abs(action.steering_curvature_change)
        lateral_dev_cost = self.params.lateral_dev_weight * obs.lateral_dev ** 2
        cost = curvature_cost + curvature_change_cost + lateral_dev_cost

        violation_buffer_overshoot = abs(obs.lateral_dev) - 4
        violation_multiplier = np.clip(1 - violation_buffer_overshoot, 0, 1)

        return violation_multiplier * self.params.survival_reward / (1 + cost)


class PathGenerator(ABC):
    @abstractmethod
    def get_path(self, ds, seed) -> np.ndarray:
        pass


class DubinsPathGenerator(PathGenerator):
    @dataclass
    class DubinsPathParams:
        num_way_points: int = 1000
        min_radius: float = 10
        max_radius: float = 50
        position_range: float = 100
        check_intersection: bool = False
        intersection_dist: float = 10

    def __init__(self, params=DubinsPathParams()):
        self.params = params

    def get_path(self, ds, seed) -> np.ndarray:
        while (True):
            waypoints = np.random.rand(self.params.num_way_points, 4)
            waypoints[:, 0:1] *= self.params.position_range
            waypoints[:, 2] *= 2 * np.pi
            waypoints[:, 3] = np.random.uniform(self.params.min_radius, self.params.max_radius,
                                                self.params.num_way_points)
            path = get_multi_waypoint_from_np(waypoints, ds)
            if not self.params.check_intersection or is_path_valid(path=path,
                                                                   min_distance=self.params.intersection_dist, step=ds):
                return path


class PathEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human", "none", "save"]}

    @dataclass
    class Params:
        ds: float = 1.0
        N_lookahead: int = 20
        N_lookahead_scale: int = 4
        max_curvature: float = 0.2
        max_curvature_change: float = 0.1
        max_deviation: float = 5
        max_deviation_buffer: float = 0.5
        random_state: bool = True
        start_at_beginning: bool = False
        reuse_path: bool = True
        max_steps: int = 500

    def __init__(self, render_mode="none", save_dir=None, params=Params()):
        super().__init__()
        self.render_mode = render_mode
        self.save_dir = save_dir
        self.params = params

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3 + self.params.N_lookahead,), dtype=np.float32
        )
        self.station = 0.0

        self.path_generator = None
        self.path = None
        self.reward_class = Reward(params=Reward.Params())

        self.new_path_to_draw = False

        if self.render_mode == "human" or self.render_mode == "save":
            self._init_fig()

    def _init_fig(self):
        self.state_plot_idx = 0
        self.map_plot_idx = 1
        self.state_plot_2_idx = 2
        self.reward_plot_idx = 3

        self.fig, self.axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        self.axs = self.axs.flatten()
        self.lines = []

        self.axs[self.state_plot_idx].set_xlabel("Station [m]")
        lateral_dev: float
        local_heading: float
        steering_curvature: float
        (lat_dev_line,) = self.axs[self.state_plot_2_idx].plot([], [], "k--", label='lateral deviation')
        (local_heading_line,) = self.axs[self.state_plot_2_idx].plot([], [], "r--", label='local heading')
        (steering_curvature_line,) = self.axs[self.state_plot_idx].plot([], [], "c-", label='steering curvature')
        (steering_curvature_change_line,) = self.axs[self.state_plot_idx].plot([], [], "b-",
                                                                               label='steering curvature change')
        (path_curvature_line,) = self.axs[self.state_plot_idx].plot([], [], "m--", label='path curvature')
        self.axs[self.state_plot_idx].legend()
        self.axs[self.state_plot_2_idx].legend()

        (line,) = self.axs[self.map_plot_idx].plot([], [], "b--")
        (right_boundary_line,) = self.axs[self.map_plot_idx].plot([], [], "k-")
        (left_boundary_line,) = self.axs[self.map_plot_idx].plot([], [], "k-")
        (ego_position_line,) = self.axs[self.map_plot_idx].plot([], [], "m-")
        (ego_box_line,) = self.axs[self.map_plot_idx].plot([], [], "b-")

        (reward_line,) = self.axs[self.reward_plot_idx].plot([], [], "-")

        self.axs[self.map_plot_idx].axis('equal')
        self.axs[self.map_plot_idx].set_xlabel("x [m]")
        self.axs[self.map_plot_idx].set_ylabel("y [m]")
        self.axs[self.reward_plot_idx].set_ylabel("Reward")

        self.lines.append({
            "steering_curvature": steering_curvature_line,
            "steering_curvature_change": steering_curvature_change_line,
            "path_curvature": path_curvature_line
        })
        self.lines.append(
            {"baseline": line, "left_boundary_line": left_boundary_line, "right_boundary_line": right_boundary_line,
             "ego_position_line": ego_position_line, "ego_box_line": ego_box_line})
        self.lines.append(
            {
                "lateral_dev": lat_dev_line,
                "local_heading": local_heading_line,
            }
        )
        self.lines.append(
            {self.lines.append({"reward": reward_line})}
        )

    def _compute_cartesian_position(self, state: State):
        path_heading = self.path[state.path_index, 2]

        # Calculate the unit vector 90 degrees to the left
        left_unit_vec = (-np.sin(path_heading), np.cos(path_heading))

        # Compute the cartesian position with lateral deviation
        x = self.path[state.path_index, 0] + left_unit_vec[0] * state.lateral_dev
        y = self.path[state.path_index, 1] + left_unit_vec[1] * state.lateral_dev

        # heading
        heading = path_heading + state.local_heading

        return x, y, heading

    def _compute_positions(self, states):
        x_values = []
        y_values = []
        heading_values = []

        for state in states:
            x, y, heading = self._compute_cartesian_position(state)
            x_values.append(x)
            y_values.append(y)
            heading_values.append(heading)

        return np.array(x_values), np.array(y_values), np.array(heading_values)

    def _render(self):
        if self.new_path_to_draw:
            self.lines[self.map_plot_idx]["baseline"].set_data(self.path[:, 0], self.path[:, 1])

            # Extract positions and headings
            x_positions = self.path[:, 0]
            y_positions = self.path[:, 1]
            headings = self.path[:, 2]

            # Calculate offsets for right and left boundary lines
            offset = 5
            right_x = x_positions + offset * np.cos(headings + np.pi / 2)
            right_y = y_positions + offset * np.sin(headings + np.pi / 2)
            left_x = x_positions - offset * np.cos(headings + np.pi / 2)
            left_y = y_positions - offset * np.sin(headings + np.pi / 2)

            # Update the right and left boundary lines
            self.lines[self.map_plot_idx]["right_boundary_line"].set_data(right_x, right_y)
            self.lines[self.map_plot_idx]["left_boundary_line"].set_data(left_x, left_y)

            # Determine axis limits to include all data points
            min_x = min(x_positions.min(), right_x.min(), left_x.min()) - 5
            max_x = max(x_positions.max(), right_x.max(), left_x.max()) + 5
            min_y = min(y_positions.min(), right_y.min(), left_y.min()) - 5
            max_y = max(y_positions.max(), right_y.max(), left_y.max()) + 5

            # Ensure equal aspect ratio by expanding the smaller range to match the larger one
            x_range = max_x - min_x
            y_range = max_y - min_y

            if x_range > y_range:
                delta = (x_range - y_range) / 2
                min_y -= delta
                max_y += delta
            else:
                delta = (y_range - x_range) / 2
                min_x -= delta
                max_x += delta

            self.axs[self.map_plot_idx].set_xlim(min_x, max_x)
            self.axs[self.map_plot_idx].set_ylim(min_y, max_y)
            self.axs[self.state_plot_idx].set_xlim(0, self.max_station)
            self.axs[self.state_plot_idx].set_ylim(-0.2, 0.2)
            self.axs[self.state_plot_2_idx].set_xlim(0, self.max_station)
            self.axs[self.state_plot_2_idx].set_ylim(-2, 2)
            self.axs[self.reward_plot_idx].set_xlim(0, self.max_station)
            self.axs[self.reward_plot_idx].set_ylim(-0.1, 1.1)
            self.new_path_to_draw = False

        # set ego path
        ego_path_x, ego_path_y, ego_path_heading = self._compute_positions(self.state_array)
        self.lines[self.map_plot_idx]["ego_position_line"].set_data(ego_path_x, ego_path_y)

        # draw car
        x_box, y_box = car_polygon(ego_path_x[-1], ego_path_y[-1], ego_path_heading[-1])
        self.lines[self.map_plot_idx]["ego_box_line"].set_data(x_box, y_box)

        # draw state
        self.lines[self.state_plot_2_idx]["lateral_dev"].set_data(self.state_station_array,
                                                                  [s.lateral_dev for s in self.state_array])
        self.lines[self.state_plot_2_idx]["local_heading"].set_data(self.state_station_array,
                                                                    [s.local_heading for s in self.state_array])
        self.lines[self.state_plot_idx]["steering_curvature"].set_data(self.state_station_array,
                                                                       [s.steering_curvature for s in self.state_array])
        self.lines[self.state_plot_idx]["steering_curvature_change"].set_data(self.state_station_array,
                                                                              [a.steering_curvature_change for a in
                                                                               self.action_array])
        self.lines[self.state_plot_idx]["path_curvature"].set_data(self.state_station_array, self.path_curvature_array)

        self.lines[self.reward_plot_idx]["reward"].set_data(
            self.state_station_array, self.reward_array
        )

        plt.draw()

    def step(self, action):
        # update state
        action = Action(steering_curvature_change=action[0] * self.params.max_curvature_change)
        self.path_curvature_array.append(self.path[self.state.path_index, 3])

        self.state = motion_model(
            state=self.state,
            action=action,
            path=self.path,
            ds=self.params.ds,
            max_steering_curvature=self.params.max_curvature
        )

        self.station += self.params.ds
        self.action_array.append(action)
        self.state_array.append(self.state)
        self.state_station_array.append(self.station)

        # get observation
        observation = Observation.build(
            state=self.state, path=self.path, max_lateral_dev=self.params.max_deviation,
            N_lookahead=self.params.N_lookahead,
            N_lookahead_scale=self.params.N_lookahead_scale
        )

        # compute reward
        reward = self.reward_class.compute_reward(observation, action)
        self.reward_array.append(reward)

        truncated, terminated = False, False

        if self.state.path_index > self.starting_index + self.params.max_steps or self.state.path_index + self.params.N_lookahead * self.params.N_lookahead_scale >= \
                self.path.shape[0]:
            truncated = True
        if abs(observation.lateral_dev) > self.params.max_deviation:
            terminated = True

        return observation.to_np(), float(reward), terminated, truncated, {}

    def reset(
            self,
            ego_init_state: Optional[State] = None,
            path_generator: Optional[PathGenerator] = None,
            seed=None,
            options=None,
    ):
        super().reset(seed=seed, options=options)
        self.station = 0.0

        if self.path is None or not self.params.reuse_path:
            if path_generator is None:
                self.path_generator = DubinsPathGenerator(params=DubinsPathGenerator.DubinsPathParams())
            else:
                self.path_generator = path_generator

            self.path = self.path_generator.get_path(self.params.ds, seed)
            self.new_path_to_draw = True

        if ego_init_state is not None:
            self.state = ego_init_state
        else:
            if self.params.random_state:
                max_lateral_dev = self.params.max_deviation - self.params.max_deviation_buffer
                min_lateral_dev = -max_lateral_dev
                self.state = State(
                    lateral_dev=np.random.rand() * (max_lateral_dev - min_lateral_dev) + min_lateral_dev,
                    path_index=np.random.randint(0, self.path.shape[
                        0] - self.params.N_lookahead * self.params.N_lookahead_scale),
                    local_heading=0,
                    steering_curvature=0
                )
                if self.params.start_at_beginning:
                    self.state.path_index = 0
            else:
                self.state = State(
                    lateral_dev=0,
                    local_heading=0,
                    steering_curvature=0,
                    path_index=0
                )

        self.action_array = [Action(steering_curvature_change=0.0)]
        self.state_array = [self.state]
        self.path_curvature_array = [0.0]
        self.state_station_array = [self.station]
        self.max_station = min(self.params.max_steps, len(self.path)) * self.params.ds
        self.reward_array = [0.0]

        self.starting_index = self.state.path_index

        # get observation
        observation = Observation.build(
            state=self.state, path=self.path, max_lateral_dev=self.params.max_deviation,
            N_lookahead=self.params.N_lookahead,
            N_lookahead_scale=self.params.N_lookahead_scale
        ).to_np()

        info = {}
        return observation, info

    def render(self, file_name=None, title=None):
        if title:
            plt.suptitle(title)
        if self.render_mode == "save" and self.time >= self.params.max_time:
            self._render()
            plt.savefig(f"{self.save_dir}/{file_name}.png")
        elif self.render_mode == "human":
            self._render()
            # figure out how to do this best
            plt.pause(0.1)

    def close(self):
        pass


def pid(obs_in: np.ndarray):
    lateral_dev = obs_in[0]
    local_heading = obs_in[1]
    steering_curvature = obs_in[2]
    path_curvature = obs_in[3 + 10]
    k = 0.6
    for i in range(9, 0, -1):
        path_curvature = k * path_curvature + (1 - k) * obs_in[3 + i]

    target_local_heading = -0.03 * lateral_dev
    target_curvature = 0.07 * (target_local_heading - local_heading) + path_curvature
    return 0.03 * (target_curvature - steering_curvature)


if __name__ == "__main__":
    env = PathEnv(render_mode="human")
    check_env(env, warn=True)

    obs, _ = env.reset()

    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())

    truncated = False
    terminated = False
    action = 0
    while truncated is False and terminated is False and plt.get_fignums():
        obs, reward, terminated, truncated, info = env.step([action])
        action = pid(obs)
        print(obs)
        print(reward)
        env.render()
    plt.show()
