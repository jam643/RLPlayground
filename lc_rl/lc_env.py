import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass, field, fields
from matplotlib import pyplot as plt
from simple_road import generate_standard_lookahead_profile, SimpleRoad, RoadGenerator, DubinsRoadGenerator, \
    FixedScenarioRoadGenerator, FixedScenario
import skimage.measure
import time
from typing import Optional, Dict, List
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
from scipy import integrate
from enum import Enum
from stable_baselines3.common.env_checker import check_env
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import timeit


@dataclass
class State:
    station: float
    lateral_dev: float
    velocity: float
    acceleration: float
    local_heading: float
    steering_angle: float


@dataclass
class DerivedStateValues:
    beta: float
    lateral_acceleration: float
    speed_along_path: float


class RenderMode(Enum):
    Null = 0
    Human = 1
    Save = 2


@dataclass
class Action:
    steering_angle_change: float
    jerk: float


class IncurringCar:
    path_index: int
    lateral_dev: float
    angle: float = 0.0
    front_length: float = 2.5
    rear_length: float = 2.5
    width: float = 2.0


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


# NOTE: the derived state values correspond to the input state, not to the next state
def motion_model(
        state: State, action: Action, max_steering_angle: float, min_acc: float, max_acc: float, l_r: float, l_f: float,
        road: SimpleRoad, dt: float
) -> (State, DerivedStateValues):
    beta = np.arctan2(l_r * np.tan(state.steering_angle), (l_r + l_f))

    # interpolate path state
    path_pose = road.get_pose_at_station(state.station)

    path_curvature = path_pose[3]
    ego_ds_per_path_ds = (1 - state.lateral_dev * path_curvature) / np.cos(state.local_heading + beta)
    lateral_dev = state.lateral_dev + np.sin(state.local_heading + beta) * state.velocity * dt

    kinematic_curvature = np.sin(beta) / l_r
    speed_along_path = state.velocity / ego_ds_per_path_ds
    station_change = speed_along_path * dt
    local_heading = state.local_heading + kinematic_curvature * state.velocity * dt - path_curvature * station_change
    station = state.station + station_change

    steering_angle = state.steering_angle + action.steering_angle_change * dt
    steering_angle = np.clip(steering_angle, -max_steering_angle, max_steering_angle)

    velocity = state.velocity + state.acceleration * dt
    velocity = max(0.0, velocity)

    acceleration = state.acceleration + action.jerk * dt
    acceleration = np.clip(acceleration, min_acc, max_acc)

    lat_acceleration = kinematic_curvature * state.velocity * state.velocity

    return State(station=station, lateral_dev=lateral_dev, velocity=velocity, acceleration=acceleration,
                 local_heading=local_heading,
                 steering_angle=steering_angle), DerivedStateValues(beta=beta, lateral_acceleration=lat_acceleration,
                                                                    speed_along_path=speed_along_path)


@dataclass
class Observation:
    lateral_dev: float
    local_heading: float
    steering_angle: float
    velocity: float
    acceleration: float

    curvature_lookahead: np.ndarray
    left_width_lookahead: np.ndarray
    right_width_lookahead: np.ndarray

    # normalization value
    max_lateral_dev: float
    speed_normalization: float
    accel_normalization: float

    @staticmethod
    def build(state: State, road: SimpleRoad, max_lateral_dev: float, max_speed: float, max_abs_acc: float,
              lookahead_profile: np.ndarray):
        curvature_lookahead = road.get_curvature_lookahead(station=state.station, lookahead_profile=lookahead_profile)
        left_width_lookahead, right_width_lookahead = road.get_width_lookahead(station=state.station,
                                                                               lookahead_profile=lookahead_profile)

        # build curvature lookahaed
        return Observation(
            lateral_dev=state.lateral_dev,
            local_heading=state.local_heading,
            steering_angle=state.steering_angle,
            velocity=state.velocity,
            acceleration=state.acceleration,
            curvature_lookahead=curvature_lookahead,
            left_width_lookahead=left_width_lookahead,
            right_width_lookahead=right_width_lookahead,
            max_lateral_dev=max_lateral_dev,
            speed_normalization=max_speed,
            accel_normalization=max_abs_acc
        )

    def to_scaled_np(self):
        state = np.array(
            [
                self.lateral_dev / self.max_lateral_dev,  # normalize observation
                self.local_heading,
                self.steering_angle,
                self.velocity / self.speed_normalization,
                self.acceleration / self.accel_normalization
            ]
        )
        scaled_left_width_lookahead = self.left_width_lookahead / self.max_lateral_dev
        scaled_right_width_lookahead = self.right_width_lookahead / self.max_lateral_dev
        full_obs = (np.concatenate(
            [state, self.curvature_lookahead, scaled_left_width_lookahead, scaled_right_width_lookahead]).astype(
            np.float32))
        return full_obs


class Reward:
    @dataclass
    class Params:
        acceleration_weight: float = 0.2
        jerk_weight: float = 3
        steering_change_weight: float = 1
        lateral_dev_weight: float = 0
        progress_reward: float = 0.2
        violation_buffer_start: int = 2
        violation_buffer_size: int = 3
        violation_cost: int = 100
        survivor_reward: float = 1.0

    def __init__(self, params: Params):
        self.params = params

    def get_empty_detail_dict(self) -> Dict[str, List[float]]:
        return {
            "acceleration_cost": [],
            "steering_change_cost": [],
            "jerk_cost": [],
            "lateral_dev_cost": [],
            "comfort_cost": [],
            "progress_reward": []
        }

    def compute_reward(self, state: State, action: Action, derived_state: DerivedStateValues, is_crashed: bool,
                       detail_dict=None) -> float:
        acceleration_cost = (
                                    derived_state.lateral_acceleration ** 2 + state.acceleration ** 2) * self.params.acceleration_weight
        steering_change_cost = self.params.steering_change_weight * action.steering_angle_change ** 2
        jerk_cost = self.params.jerk_weight * action.jerk ** 2
        lateral_dev_cost = self.params.lateral_dev_weight * state.lateral_dev ** 2
        comfort_cost = acceleration_cost + lateral_dev_cost + steering_change_cost + jerk_cost
        progress_reward = derived_state.speed_along_path * self.params.progress_reward

        if detail_dict is not None:
            detail_dict["acceleration_cost"].append(acceleration_cost)
            detail_dict["steering_change_cost"].append(steering_change_cost)
            detail_dict["jerk_cost"].append(jerk_cost)
            detail_dict["lateral_dev_cost"].append(lateral_dev_cost)
            detail_dict["progress_reward"].append(progress_reward)

        return progress_reward - comfort_cost + self.params.survivor_reward


@dataclass
class ScenarioOptionProbabilities:
    random_roads_no_lead: float = 0.5
    straight_road_no_lead: float = 0.5

    def __postinit__(self):
        sum_probs = np.sum([getattr(self, attr.name) for attr in fields(self)])
        assert (
                sum_probs == 1.0
        ), f"Probabilities must sum to 1.0, but sum is {sum_probs}"

    def sample_scenario(self):
        return np.random.choice(
            list(self.__dict__.keys()),
            p=list(self.__dict__.values()),
        )


class LcEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human", "none", "save"]}

    @dataclass
    class Params:
        path_ds: float = 1.0
        time_dt: float = 0.2
        max_steering_angle: float = 0.54
        max_steering_angle_change: float = 0.2
        l_r: float = 1.3634
        l_f: float = 1.6366
        overhang: float = 0.8
        lookahead_length: float = 200.0
        max_deviation_buffer: float = 0.5
        random_state: bool = True
        start_at_beginning: bool = True
        max_time: float = 120.0
        min_jerk: float = -1.0
        max_jerk: float = 1.0
        max_accel: float = 2.0
        min_accel: float = -7.0
        max_combined_accel: float = 8.0
        max_expected_speed: float = 20.0
        max_expected_deviation: float = 2.0

    def __init__(self, render_mode="none", params=Params(), lookahead_profile: np.ndarray = None):
        super().__init__()
        self.render_mode = render_mode
        self.params = params
        self.lookahead_profile = lookahead_profile if lookahead_profile is not None else generate_standard_lookahead_profile()
        self.reward_detail_dict = None

        self.num_road_generator_called = 0

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32,
        )

        lookahead_length = len(self.lookahead_profile)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5 + 3 * lookahead_length,), dtype=np.float32
        )

        self.road_generator = None
        self.road = None
        self.reward_class = Reward(params=Reward.Params())

        self.new_path_to_draw = False

        self.left_incurring_cars = []
        self.right_incurring_cars = []

        if self.render_mode == "human" or self.render_mode == "save":
            self._init_fig()

    def _init_fig(self):
        self.state_plot_idx = 0
        self.map_plot_idx = 1
        self.speed_plot_idx = 2
        self.reward_plot_idx = 3

        self.fig, self.axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        self.axs = self.axs.flatten()
        self.lines = []

        self.axs[self.state_plot_idx].set_xlabel("Time [s]")
        self.axs[self.speed_plot_idx].set_xlabel("Time [s]")
        lateral_dev: float
        local_heading: float
        steering_curvature: float
        (speed_plot_line,) = self.axs[self.speed_plot_idx].plot([], [], "k--", label='speed')
        (acc_plot_line,) = self.axs[self.speed_plot_idx].plot([], [], "r--", label='longitudinal acceleration')
        (lat_acc_plot_line,) = self.axs[self.speed_plot_idx].plot([], [], "c--", label='lateral acceleration')
        (steering_curvature_line,) = self.axs[self.state_plot_idx].plot([], [], "c-", label='steering angle')
        (steering_curvature_change_line,) = self.axs[self.state_plot_idx].plot([], [], "b-",
                                                                               label='steering angle change')
        (path_curvature_line,) = self.axs[self.state_plot_idx].plot([], [], "m--", label='path curvature')
        self.axs[self.state_plot_idx].legend()
        self.axs[self.speed_plot_idx].legend()

        (line,) = self.axs[self.map_plot_idx].plot([], [], "b--")
        (right_boundary_line,) = self.axs[self.map_plot_idx].plot([], [], "k-")
        (left_boundary_line,) = self.axs[self.map_plot_idx].plot([], [], "k-")
        (ego_position_line,) = self.axs[self.map_plot_idx].plot([], [], "m-")
        (ego_box_line,) = self.axs[self.map_plot_idx].plot([], [], "b-")

        (reward_line,) = self.axs[self.reward_plot_idx].plot([], [], "-", label="reward")
        (acceleration_cost_line,) = self.axs[self.reward_plot_idx].plot([], [], "c--", label="acceleration_cost")
        (steering_change_cost_line,) = self.axs[self.reward_plot_idx].plot([], [], "m--", label="steering_change_cost")
        (jerk_cost_line,) = self.axs[self.reward_plot_idx].plot([], [], "g--", label="jerk_cost")
        (lateral_dev_cost_line,) = self.axs[self.reward_plot_idx].plot([], [], "r--", label="lateral_dev_cost")
        (progress_reward_line,) = self.axs[self.reward_plot_idx].plot([], [], "y--", label="progress_reward")
        self.axs[self.reward_plot_idx].legend()

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
                "speed": speed_plot_line,
                "acc": acc_plot_line,
                "lat_acc": lat_acc_plot_line,
            }
        )
        self.lines.append(
            {
                "reward": reward_line,
                "acceleration_cost": acceleration_cost_line,
                "steering_change_cost": steering_change_cost_line,
                "jerk_cost": jerk_cost_line,
                "lateral_dev_cost": lateral_dev_cost_line,
                "progress_reward": progress_reward_line
            }
        )

        # colormap
        self.norm = mcolors.Normalize(vmin=self.params.min_accel, vmax=self.params.max_accel)
        self.cmap = mcolors.LinearSegmentedColormap.from_list('rg', ["r", "b", "g"], N=256)

    def _compute_cartesian_position(self, state: State):
        path_pose = self.road.get_pose_at_station(state.station)  #TODOOOOOOOOOO
        path_heading = path_pose[2]

        # Calculate the unit vector 90 degrees to the left
        left_unit_vec = (-np.sin(path_heading), np.cos(path_heading))

        # Compute the cartesian position with lateral deviation
        x = path_pose[0] + left_unit_vec[0] * state.lateral_dev
        y = path_pose[1] + left_unit_vec[1] * state.lateral_dev

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
            baseline_x, baseline_y, left_x, left_y, right_x, right_y = self.road.get_lines_for_visualization()

            self.lines[self.map_plot_idx]["baseline"].set_data(baseline_x, baseline_y)
            self.lines[self.map_plot_idx]["right_boundary_line"].set_data(right_x, right_y)
            self.lines[self.map_plot_idx]["left_boundary_line"].set_data(left_x, left_y)

            # Determine axis limits to include all data points
            min_x = min(baseline_x.min(), right_x.min(), left_x.min()) - 5
            max_x = max(baseline_x.max(), right_x.max(), left_x.max()) + 5
            min_y = min(baseline_y.min(), right_y.min(), left_y.min()) - 5
            max_y = max(baseline_y.max(), right_y.max(), left_y.max()) + 5

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
            self.new_path_to_draw = False

        min_time_scale = 20.0

        self.axs[self.state_plot_idx].set_xlim(0, max(min_time_scale, self.time_array[-1]))
        self.axs[self.state_plot_idx].set_ylim(-0.2, 0.2)
        self.axs[self.speed_plot_idx].set_xlim(0, max(min_time_scale, self.time_array[-1]))
        self.axs[self.speed_plot_idx].set_ylim(-11, 21)
        self.axs[self.reward_plot_idx].set_xlim(0, max(min_time_scale, self.time_array[-1]))
        self.axs[self.reward_plot_idx].set_ylim(-0.1, 2.1)

        # set ego path
        ego_path_x, ego_path_y, ego_path_heading = self._compute_positions(self.state_array)
        self.lines[self.map_plot_idx]["ego_position_line"].set_data(ego_path_x, ego_path_y)
        #colors = self.cmap(self.norm([s.acceleration for s in self.state_array]))
        #self.lines[self.map_plot_idx]["ego_position_line"].set_color(colors)

        # draw car
        x_box, y_box = car_polygon(ego_path_x[-1], ego_path_y[-1], ego_path_heading[-1])
        self.lines[self.map_plot_idx]["ego_box_line"].set_data(x_box, y_box)

        # draw state
        self.lines[self.speed_plot_idx]["speed"].set_data(self.time_array,
                                                          [s.velocity for s in self.state_array])
        self.lines[self.speed_plot_idx]["acc"].set_data(self.time_array,
                                                        [s.acceleration for s in self.state_array])
        self.lines[self.speed_plot_idx]["lat_acc"].set_data(self.time_array[:-1],
                                                            [s.lateral_acceleration for s in self.derived_state_array])
        self.lines[self.state_plot_idx]["steering_curvature"].set_data(self.time_array,
                                                                       [s.steering_angle for s in self.state_array])
        self.lines[self.state_plot_idx]["steering_curvature_change"].set_data(self.time_array,
                                                                              [a.steering_angle_change for a in
                                                                               self.action_array])
        self.lines[self.state_plot_idx]["path_curvature"].set_data(self.time_array, self.path_curvature_array)

        self.lines[self.reward_plot_idx]["reward"].set_data(
            self.time_array, self.reward_array
        )

        self.lines[self.reward_plot_idx]["acceleration_cost"].set_data(self.time_array[:-1],
                                                                       self.reward_detail_dict["acceleration_cost"])
        self.lines[self.reward_plot_idx]["steering_change_cost"].set_data(self.time_array[:-1], self.reward_detail_dict[
            "steering_change_cost"])
        self.lines[self.reward_plot_idx]["jerk_cost"].set_data(self.time_array[:-1],
                                                               self.reward_detail_dict["jerk_cost"])
        self.lines[self.reward_plot_idx]["lateral_dev_cost"].set_data(self.time_array[:-1],
                                                                      self.reward_detail_dict["lateral_dev_cost"])
        self.lines[self.reward_plot_idx]["progress_reward"].set_data(self.time_array[:-1],
                                                                     self.reward_detail_dict["progress_reward"])

        plt.draw()

    def _is_crashed(self):
        left_width, right_width = self.road.get_width_at_station(self.state.station)
        overrotated = abs(self.state_array[-1].local_heading) > 1.5
        if len(self.derived_state_array) > 0:
            combined_acc_sqr = self.derived_state_array[-1].lateral_acceleration ** 2 + self.state_array[
                -2].acceleration ** 2
        return overrotated or self.state.lateral_dev > left_width or self.state.lateral_dev < -right_width or combined_acc_sqr > self.params.max_combined_accel ** 2

    def step(self, action):
        # update state
        steering_angle_change = action[0] * self.params.max_steering_angle_change
        jerk = action[1] * (self.params.max_jerk - self.params.min_jerk) / 2.0 + (
                self.params.max_jerk + self.params.min_jerk) / 2.0
        action = Action(steering_angle_change=steering_angle_change, jerk=jerk)
        path_pose = self.road.get_pose_at_station(self.state.station)
        self.path_curvature_array.append(path_pose[3])

        self.state, derived_state = motion_model(
            state=self.state,
            action=action,
            max_steering_angle=self.params.max_steering_angle,
            min_acc=self.params.min_accel,
            max_acc=self.params.max_accel,
            l_r=self.params.l_r,
            l_f=self.params.l_f,
            road=self.road,
            dt=self.params.time_dt
        )

        self.action_array.append(action)
        self.state_array.append(self.state)
        self.derived_state_array.append(derived_state)
        self.time_array.append(self.time_array[-1] + self.params.time_dt)

        is_crashed = self._is_crashed()
        # compute reward
        reward = self.reward_class.compute_reward(state=self.state, derived_state=self.derived_state_array[-1],
                                                  action=action, is_crashed=is_crashed,
                                                  detail_dict=self.reward_detail_dict)
        self.reward_array.append(reward)

        truncated, terminated = False, False

        if self.time_array[
            -1] > self.params.max_time or self.state.station + self.params.lookahead_length >= self.road.get_max_station():
            truncated = True
        if is_crashed:
            terminated = True

        # get observation
        observation = Observation.build(state=self.state, road=self.road,
                                        max_lateral_dev=self.params.max_expected_deviation,
                                        max_abs_acc=-self.params.min_accel,  # TODO: maybe better value here
                                        max_speed=self.params.max_expected_speed,
                                        lookahead_profile=self.lookahead_profile).to_scaled_np()

        return observation, float(reward), terminated, truncated, {}

    def reset(
            self,
            ego_init_state: Optional[State] = None,
            road_generators: Optional[Dict] = None,
            scenario_option_probabilities: Optional[ScenarioOptionProbabilities] = None,
            road: Optional[SimpleRoad] = None,
            seed=None,
            options=None,
    ):
        super().reset(seed=seed, options=options)
        self.steps = 0

        if scenario_option_probabilities is None:
            scenario_option_probabilities = ScenarioOptionProbabilities()
        scenario_choice = scenario_option_probabilities.sample_scenario()

        # assert
        assert road is None or road_generators is None

        if road is not None:
            self.road = road
        else:
            if road_generators is not None and scenario_choice in road_generators:
                self.road_generator = road_generators[scenario_choice]
            else:
                if scenario_choice == "random_roads_no_lead":
                    self.road_generator = DubinsRoadGenerator()
                if scenario_choice == "straight_road_no_lead":
                    self.road_generator = FixedScenarioRoadGenerator(scenario=FixedScenario.StraightRoad)

            self.road = self.road_generator.get_road(seed)
        self.new_path_to_draw = True

        if ego_init_state is not None:
            self.state = ego_init_state
        else:
            if self.params.random_state:
                max_speed = 1.0

                station = 0
                if not self.params.start_at_beginning:
                    station: float = np.random.rand() * (self.road.get_max_station() - self.params.lookahead_length)

                left_width, right_width = self.road.get_width_at_station(station)
                mid_point = 0.5 * left_width - 0.5 * right_width
                lateral_dev: float
                velocity: float
                acceleration: float
                local_heading: float
                steering_angle: float

                self.state = State(
                    station=station,
                    lateral_dev=mid_point,
                    velocity=np.random.rand() * max_speed,
                    acceleration=0.0,
                    local_heading=0.0,
                    steering_angle=0.0
                )
            else:
                self.state = State(
                    station=0.0,
                    lateral_dev=0.0,
                    velocity=0.0,
                    acceleration=0.0,
                    local_heading=0.0,
                    steering_angle=0.0
                )

        self.action_array = [Action(steering_angle_change=0.0, jerk=0.0)]
        self.state_array = [self.state]
        self.derived_state_array = []
        self.time_array = [0.0]
        self.path_curvature_array = [0.0]
        self.reward_array = [0.0]

        if self.render_mode == "human" or self.render_mode == "save":
            self.reward_detail_dict = self.reward_class.get_empty_detail_dict()

        # get observation
        observation = Observation.build(state=self.state, road=self.road,
                                        max_lateral_dev=self.params.max_expected_deviation,
                                        max_abs_acc=-self.params.min_accel,
                                        max_speed=self.params.max_expected_speed,
                                        lookahead_profile=self.lookahead_profile).to_scaled_np()

        info = {}
        return observation, info

    def render(self, save_dir=None, file_name=None, title=None):
        if title:
            plt.suptitle(title)
        if self.render_mode == "save":
            self._render()
            plt.savefig(f"{save_dir}/{file_name}.png")
        elif self.render_mode == "human":
            self._render()
            # figure out how to do this best
            plt.pause(0.02)

    def close(self):
        pass


def pid(obs_in: np.ndarray):
    lateral_dev = obs_in[0]
    local_heading = obs_in[1]
    steering_curvature = obs_in[2]
    speed = obs_in[3] * 20
    accel = obs_in[4] * 7
    path_curvature = obs_in[5 + 10]
    k = 0.7
    for i in range(9, 0, -1):
        path_curvature = k * path_curvature + (1 - k) * obs_in[5 + i]

    target_speed = np.sqrt(1.0 / (0.01 + abs(path_curvature)))

    target_accel = 0.5 * (target_speed - speed)

    target_local_heading = -0.2 * lateral_dev
    target_curvature = path_curvature + 1.0 * (target_local_heading - local_heading)
    return [3 * (target_curvature - steering_curvature) * speed, 0.1 * (target_accel - accel)]


if __name__ == "__main__":
    import time

    params_for_visualization = LcEnv.Params(start_at_beginning=False)
    env = LcEnv(render_mode="human", params=params_for_visualization)
    check_env(env, warn=True)


    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())

    prob = ScenarioOptionProbabilities(random_roads_no_lead=1.0, straight_road_no_lead=0)

    action = np.array([0.0, 0.0])
    while True:
        obs, _ = env.reset(scenario_option_probabilities=prob)
        truncated = False
        terminated = False
        while truncated is False and terminated is False and plt.get_fignums():
            obs, reward, terminated, truncated, info = env.step(action)
            action = pid(obs)
            #print(obs)
            #print(reward)
            env.render()
