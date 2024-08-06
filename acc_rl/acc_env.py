import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass, fields
from matplotlib import pyplot as plt
import time
from typing import Optional, Tuple
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
from scipy import integrate
from enum import Enum
from abc import ABC, abstractmethod
from stable_baselines3.common.env_checker import check_env


class RenderMode(Enum):
    Null = 0
    Human = 1
    Save = 2


@dataclass
class State:
    station: float
    speed: float
    acceleration: float


@dataclass
class LeadState:
    station: float
    speed: float


@dataclass
class Action:
    jerk: float


@dataclass
class LeadAction:
    acceleration: float


@dataclass
class Goal:
    station: float


def motion_model(
        state: State, action: Action, max_accel: float, min_accel: float, dt: float
) -> Tuple[State, Action]:
    acceleration = state.acceleration + action.jerk * dt
    acceleration = np.clip(acceleration, min_accel, max_accel)
    speed = state.speed + state.acceleration * dt + 0.5 * action.jerk * dt ** 2
    if speed < 0.0:
        speed = 0.0
        acceleration = 0.0
        if state.acceleration < 0.0 and state.speed > 0.0:
            action.jerk = 2 * state.speed / (dt ** 2)
    station = max(
        state.station,
        (
                state.station
                + state.speed * dt
                + state.acceleration * dt ** 2 / 2
                + 1 / 6 * action.jerk * dt ** 3
        ),
    )
    return State(station=station, speed=speed, acceleration=acceleration), action


def motion_model_lead(state: LeadState, action: LeadAction, dt: float) -> LeadState:
    speed = state.speed + action.acceleration * dt
    if speed < 0:
        speed = 0
        action.acceleration = 0
    station = state.station + speed * dt
    return LeadState(station=station, speed=speed)


def huber_loss(error: float, delta: float) -> float:
    """
    Calculate the Huber loss for a given error and delta.

    Parameters:
    error (float): The error for which the loss is to be calculated.
    delta (float): The delta value used in the Huber loss calculation.

    Returns:
    float: The calculated Huber loss.
    """
    abs_error = np.abs(error)
    if abs_error <= delta:
        loss = 0.5 * error ** 2
    else:
        loss = delta * (abs_error - 0.5 * delta)
    return loss


class LeadCarModel(ABC):
    @dataclass
    class Params:
        dt: float

    def __init__(self, params: Params, init_state: LeadState):
        self.params = params
        self.state = init_state

    @abstractmethod
    def get_action(self, time: float, state: LeadState) -> LeadAction:
        pass

    def step(self, time: float, ego_state: State = State(0, 0, 0)):
        action = self.get_action(time, self.state)
        self.state = motion_model_lead(self.state, action, self.params.dt)
        return self.state, action

    @abstractmethod
    def does_lead_exist(self, time) -> bool:
        pass


class ConstSpeedLeadModel(LeadCarModel):
    def get_action(self, time: float, state: LeadState) -> LeadAction:
        return LeadAction(acceleration=0.0)

    def does_lead_exist(self, time) -> bool:
        return True


class NoLeadModel(LeadCarModel):
    def __init__(self):
        super().__init__(params=LeadCarModel.Params(dt=0.0), init_state=LeadState(0, 0))

    def get_action(self, time: float, state: LeadState) -> LeadAction:
        return LeadAction(acceleration=0.0)

    def step(self, time: float, ego_state: State = State(0, 0, 0)):
        return LeadState(0, 0), LeadAction(0.0)

    def does_lead_exist(self, time: float) -> bool:
        return False


class DecelLeadModel(LeadCarModel):
    @dataclass
    class DecelParams:
        time_accel_start: float
        accel: float
        accel_duration: float

    def __init__(
            self,
            params: LeadCarModel.Params,
            init_state: LeadState,
            decel_params: DecelParams,
    ):
        super().__init__(params, init_state)
        self.decel_params = decel_params

    def get_action(self, time: float, state: LeadState) -> LeadAction:
        action = LeadAction(acceleration=0.0)
        if (
                self.decel_params.time_accel_start <= time < self.decel_params.time_accel_start + self.decel_params.accel_duration
        ):
            action = LeadAction(acceleration=self.decel_params.accel)
        return action

    def does_lead_exist(self, time: float) -> bool:
        return True


class CutinLeadModel(LeadCarModel):
    @dataclass
    class CutinParams:
        time_cutin: float
        cutin_distance: float
        relative_speed: float

    def __init__(
            self,
            params: LeadCarModel.Params,
            init_state: LeadState,
            cutin_params: CutinParams,
    ):
        super().__init__(params, init_state)
        self.cutin_params = cutin_params

    def get_action(self, time: float, state: LeadState) -> LeadAction:
        return LeadAction(acceleration=0.0)

    def step(self, time: float, ego_state: State):
        if time < self.cutin_params.time_cutin:
            self.state = LeadState(station=ego_state.station + self.cutin_params.cutin_distance,
                                   speed=ego_state.speed + self.cutin_params.relative_speed)
            return self.state, LeadAction(0.0)
        else:
            return super().step(time)

    def does_lead_exist(self, time: float) -> bool:
        if time < self.cutin_params.time_cutin:
            return False
        else:
            return True


@dataclass
class Observation:
    relative_station: float
    relative_speed: float
    # relative_accel: float
    ego_speed: float
    ego_acceleration: float
    desired_speed_error: float
    desired_station_error: float
    # lead_speed: float
    # lead_accel: float
    # ttc: float
    is_lead: float
    is_goal_far: float

    @staticmethod
    def build(
            ego_state: State,
            lead_state: LeadState,
            lead_action: LeadAction,
            desired_station: float,
            desired_speed: float,
            lead_car_model: LeadCarModel,
            time: float,
    ):
        relative_station = lead_state.station - ego_state.station
        lead_speed = lead_state.speed
        relative_accel = lead_action.acceleration - ego_state.acceleration
        relative_speed = lead_state.speed - ego_state.speed
        lead_accel = lead_action.acceleration
        # ttc = Observation.compute_ttc_est(ego_state, lead_state, lead_action)
        # if np.isnan(ttc) or ttc > 10.0:
        #     ttc = 10.0
        is_lead = 1.0
        is_goal_far = 0.0
        if not lead_car_model.does_lead_exist(time):
            relative_station = 80.0
            relative_speed = 0.0
            relative_accel = 0.0
            # lead_speed = 20.0
            # lead_accel = 0.0
            # ttc = 10.0
            is_lead = 0.0

        # If we are (closer to the goal than the lead) AND < 200m from goal, or there is no lead and we are less than
        # 200 meters from the goal
        # if (desired_station - ego_state.station) <= 200 and ((lead_car_model.does_lead_exist(time) and (lead_car_model.state.station > desired_station)) or not lead_car_model.does_lead_exist(time)):
        #     is_goal_constrained = 1.0

        if desired_station - ego_state.station >= 200.0:
            is_goal_far = 1.0

        speed_lookahead = []
        for idx in range(400):
            # 200 m lookahead
            if ego_state.station + (idx/2.0) < (desired_station - 0.5):
                speed_lookahead.append(desired_speed)
            else:
                speed_lookahead.append(0)

        if ego_state.station >= desired_station:
            desired_speed = 0

        # print("Station: {}, goal: {}", ego_state.station, desired_station)
        return Observation(
            relative_station=relative_station,
            relative_speed=relative_speed,
            # relative_accel=relative_accel,
            ego_speed=ego_state.speed,
            ego_acceleration=ego_state.acceleration,
            desired_speed_error=(ego_state.speed - desired_speed),
            desired_station_error=max(-200.0, ego_state.station - desired_station),
            # lead_speed=lead_speed,
            # lead_accel=lead_accel,
            # ttc=ttc,
            is_lead=is_lead,
            # is_goal_constrained=is_goal_constrained,
            is_goal_far=is_goal_far
        )

    @staticmethod
    def solve_quadratic_real(a: float, b: float, c: float) -> float:
        radicand = b ** 2 - 4 * a * c
        if radicand < 0 or a == 0:
            return np.nan
        t1 = (-b + np.sqrt(radicand)) / (2 * a)
        t2 = (-b - np.sqrt(radicand)) / (2 * a)
        if t1 < 0:
            t1 = np.nan
        if t2 < 0:
            t2 = np.nan
        return np.nanmin([t1, t2])

    @staticmethod
    def compute_ttc_est(
            state: State, lead_state: LeadState, lead_action: LeadAction
    ) -> float:
        a = 0.5 * (lead_action.acceleration - state.acceleration)
        b = lead_state.speed - state.speed
        c = lead_state.station - state.station

        ttc_moving = np.nan
        if a == 0 and b < 0 and c > 0:
            ttc_moving = -c / b
        else:
            ttc_moving = Observation.solve_quadratic_real(a, b, c)
            if not np.isnan(ttc_moving):
                v_lead = lead_state.speed + lead_action.acceleration * ttc_moving
                v_ego = state.speed + state.acceleration * ttc_moving
                if v_lead < 0 or v_ego < 0:
                    ttc_moving = np.nan

        ttc_stopped = np.nan
        lead_stopping_distance = np.nan
        if lead_state.speed == 0 and lead_action.acceleration <= 0:
            lead_stopping_distance = lead_state.station - state.station
        elif lead_action.acceleration < 0:
            lead_stopping_distance = (
                    -0.5 * lead_state.speed ** 2 / lead_action.acceleration
                    + (lead_state.station - state.station)
            )

        if not np.isnan(lead_stopping_distance) and lead_stopping_distance >= 0:
            a = 0.5 * (-state.acceleration)
            b = -state.speed
            c = -lead_stopping_distance

            if a == 0 and b < 0 and c > 0:
                ttc_stopped = -c / b
            else:
                ttc_stopped = Observation.solve_quadratic_real(a, b, c)

        ttc = np.nanmin([ttc_moving, ttc_stopped])
        return ttc

    def to_np(self):

        arr = np.array(
            [
                self.relative_station / 40.0,
                self.relative_speed / 10.0,
                # self.relative_accel / 2.0,
                self.ego_speed / 10.0,
                self.ego_acceleration / 2.0,
                self.desired_speed_error / 10.0,
                self.desired_station_error / 40.0,
                self.is_lead,
                self.is_goal_far
            ]
        )
        #arr = np.append(arr, self.speed_lookahead)
        return arr.astype(np.float32)
        # return np.array([getattr(self, field.name) for field in fields(self)]).astype(
        #     np.float32
        # )


class Reward:
    @dataclass
    class Params:
        speed_weight: float = 0.05
        accel_weight: float = 0.2
        jerk_weight: float = 0.02
        clearance_weight: float = 10.0
        collision_weight: float = 10.0
        stationary_lead_buffer_weight: float = 0.2
        clearance_buffer_m: float = 3.0
        stationary_lead_buffer_m: float = 8.0
        goal_cost: float = 0.01

    def __init__(self, params: Params):
        self.params = params

    def compute_reward(
            self, obs: Observation, action: Action, lead_car_model: LeadCarModel, time: float
    ):
        comfort_accel_cost = self.params.accel_weight * obs.ego_acceleration ** 2

        comfort_cost = comfort_accel_cost + self.params.jerk_weight * action.jerk ** 2

        tracking_cost = self.params.speed_weight * np.abs(obs.desired_speed_error)

        tracking_cost -= 0.1 if np.abs(obs.desired_speed_error) < 0.5 else 0.0

        safety_cost = 0
        if self.is_collision(obs, lead_car_model, time) and obs.relative_speed <= 0:
            safety_cost = (
                    self.params.collision_weight
                    * obs.relative_speed ** 2
                # * huber_loss(error=(obs.lead_speed - obs.ego_speed), delta=4) ** 2
            )

        # Linearly increase if we are positive, negative (reward) if we are between 3 meters away and 0, else nothing
        goal_cost = (obs.desired_station_error * self.params.goal_cost) if obs.desired_station_error > 0 else 0.0
        clearance_cost = 0
        if obs.relative_station <= self.params.clearance_buffer_m and lead_car_model.does_lead_exist(time):
            clearance_cost = (
                    self.params.clearance_weight
                    * (obs.relative_station - self.params.clearance_buffer_m) ** 2
            )

        # stationary_lead_buffer_cost = 0
        # if (
        #     obs.relative_speed == 0.0
        #     and obs.ego_speed == 0.0
        #     and obs.relative_station >= self.params.stationary_lead_buffer_m
        #     and not isinstance(lead_car_model, NoLeadModel)
        # ):
        #     stationary_lead_buffer_cost = self.params.stationary_lead_buffer_weight * (
        #         obs.relative_station - self.params.stationary_lead_buffer_m
        #     )

        stationary_lead_buffer_reward = 0
        if (
                obs.relative_speed + obs.ego_speed <= 1e-3
                and obs.relative_station <= self.params.stationary_lead_buffer_m
                and obs.relative_station >= self.params.clearance_buffer_m
                and lead_car_model.does_lead_exist(time)
        ):
            stationary_lead_buffer_reward = self.params.speed_weight * (
                    -obs.desired_speed_error - obs.ego_speed
            )

        return (
                       -comfort_cost
                       - tracking_cost
                       - goal_cost
                       - safety_cost
                       - clearance_cost
                       + stationary_lead_buffer_reward
                   # - stationary_lead_buffer_cost
               ) / 30.0

    def is_collision(self, obs: Observation, lead_car_model: LeadCarModel, time: float):
        return obs.relative_station <= 0.0 and lead_car_model.does_lead_exist(time)


@dataclass
class ScenarioOptionProbabilities:
    decel: float = 0.0
    const_speed_lead: float = 0.0
    stationary_lead: float = 0.0
    no_lead: float = 0.0
    no_lead_from_standstill: float = 0.0

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


class ACCEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    @dataclass
    class Params:
        dt: float = 0.25
        max_time: float = 30.0
        min_jerk: float = -10.0
        max_jerk: float = 10.0
        max_accel: float = 2.0
        min_accel: float = -7.0
        speed_limit: float = 20.0
        desired_speed: float = speed_limit
        desired_station: float = 100.0

    def __init__(self, render_mode=RenderMode.Null, params=Params()):
        super().__init__()
        self.render_mode = render_mode
        self.params = params

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32,
        )
        # self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(fields(Observation)),),
            dtype=np.float32,
        )
        self.time = 0.0

        self.lead_car_model = NoLeadModel()
        self.reward_class = Reward(params=Reward.Params())

    def _init_fig(self):
        self.station_plot_idx = 0
        self.speed_plot_idx = 1
        self.acceration_plot_idx = 2
        self.jerk_plot_idx = 3
        self.reward_plot_idx = 4
        self.headway_time_plot_idx = 5
        self.ttc_plot_idx = 6
        self.rel_station_plot_idx = 7
        self.goal_station_idx = 8

        self.fig, self.axs = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
        plt.subplots_adjust(
            left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2
        )
        self.axs = self.axs.flatten()
        self.lines = []
        for ax in self.axs:
            ax.set_xlim(0, self.params.max_time)
            (line,) = ax.plot([], [], ".-")
            (line_lead,) = ax.plot([], [], ".-")
            self.lines.append({"ego": line, "lead": line_lead})
            ax.grid()
            ax.set_xlabel("Time [s]")

        self.axs[self.station_plot_idx].set_ylabel("Station [m]")
        self.axs[self.speed_plot_idx].set_ylabel("Speed [m/s]")
        self.axs[self.acceration_plot_idx].set_ylabel("Acceleration [m/s^2]")
        self.axs[self.jerk_plot_idx].set_ylabel("Jerk [m/s^3]")
        self.axs[self.reward_plot_idx].set_ylabel("Reward")
        self.axs[self.headway_time_plot_idx].set_ylabel("Following Time [s]")
        self.axs[self.ttc_plot_idx].set_ylabel("TTC [s]")
        self.axs[self.rel_station_plot_idx].set_ylabel("Relative Station [m]")
        self.axs[self.goal_station_idx].set_ylabel("Distance to goal")

        self.axs[self.speed_plot_idx].axhline(
            y=self.params.desired_speed,
            color="r",
            linestyle="--",
            label="Desired Speed",
        )
        self.axs[self.speed_plot_idx].legend(
            loc="center right", bbox_to_anchor=(1.0, 0.5)
        )

        self.axs[self.station_plot_idx].legend(
            ["Ego", "Lead"],
            loc="center right",
            bbox_to_anchor=(1.0, 0.5),
        )

    def _render(self):
        self.lines[self.speed_plot_idx]["ego"].set_data(
            self.state_time_array, [s.speed for s in self.state_array]
        )
        self.lines[self.station_plot_idx]["ego"].set_data(
            self.state_time_array, [s.station for s in self.state_array]
        )
        self.lines[self.acceration_plot_idx]["ego"].set_data(
            self.state_time_array, [a.acceleration for a in self.state_array]
        )
        self.lines[self.jerk_plot_idx]["ego"].set_data(
            self.action_time_array, [a.jerk for a in self.action_array]
        )
        self.lines[self.reward_plot_idx]["ego"].set_data(
            self.action_time_array, self.reward_array
        )
        self.lines[self.headway_time_plot_idx]["ego"].set_data(
            self.state_time_array,
            [
                (
                    ((lead_state.station - state.station) / state.speed)
                    if state.speed != 0
                    else np.nan
                )
                for state, lead_state in zip(self.state_array, self.lead_state_array)
            ],
        )
        self.lines[self.rel_station_plot_idx]["ego"].set_data(
            self.state_time_array,
            [
                (lead_state.station - state.station)
                for state, lead_state in zip(self.state_array, self.lead_state_array)
            ],
        )
        self.lines[self.goal_station_idx]["ego"].set_data(
            self.state_time_array,
            [
                (state.station - self.params.desired_station)
                for state in self.state_array
            ],
        )
        ttc_est = [
            Observation.compute_ttc_est(state, lead_state, lead_action)
            for state, lead_state, lead_action in zip(
                self.state_array[1:],
                self.lead_state_array[1:],
                self.lead_action_array,
            )
        ]
        self.lines[self.ttc_plot_idx]["ego"].set_data(
            self.action_time_array,
            ttc_est,
        )

        self.lines[self.speed_plot_idx]["lead"].set_data(
            self.state_time_array, [s.speed if self.lead_car_model.does_lead_exist(t) else np.nan for t, s in
                                    zip(self.state_time_array, self.lead_state_array)]
        )
        self.lines[self.station_plot_idx]["lead"].set_data(
            self.state_time_array, [s.station if self.lead_car_model.does_lead_exist(t) else np.nan for t, s in
                                    zip(self.state_time_array, self.lead_state_array)]
        )
        self.lines[self.acceration_plot_idx]["lead"].set_data(
            self.action_time_array, [a.acceleration if self.lead_car_model.does_lead_exist(t) else np.nan for t, a in
                                     zip(self.action_time_array, self.lead_action_array)]
        )

        self.axs[self.reward_plot_idx].legend(
            [
                f"Reward: {np.nansum(self.reward_array):.3f}",
            ],
            loc="center right",
            bbox_to_anchor=(1.0, 0.5),
        )

        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()

        self.axs[self.acceration_plot_idx].set_ylim(
            self.params.min_accel, self.params.max_accel
        )
        self.axs[self.jerk_plot_idx].set_ylim(
            self.params.min_jerk, self.params.max_jerk
        )
        self.axs[self.speed_plot_idx].set_ylim(0.0, self.params.speed_limit + 1.0)
        self.axs[self.headway_time_plot_idx].set_ylim(0.0, 10.0)
        self.axs[self.ttc_plot_idx].set_ylim(0.0, 10.0)

        plt.draw()

    def step(self, action):
        # update state
        # if action == 0:
        #     action = Action(jerk=-7.0)
        # elif action == 1:
        #     action = Action(jerk=-3.0)
        # elif action == 2:
        #     action = Action(jerk=-1.0)
        # elif action == 3:
        #     action = Action(jerk=-0.5)
        # elif action == 4:
        #     action = Action(jerk=0.0)
        # elif action == 5:
        #     action = Action(jerk=0.5)
        # elif action == 6:
        #     action = Action(jerk=1.0)
        # elif action == 7:
        #     action = Action(jerk=2.0)

        action = Action(
            jerk=action[0] * (self.params.max_jerk - self.params.min_jerk) / 2.0
                 + (self.params.max_jerk + self.params.min_jerk) / 2.0
        )
        self.state, action = motion_model(
            self.state,
            action,
            min_accel=self.params.min_accel,
            max_accel=self.params.max_accel,
            dt=self.params.dt,
        )
        self.action_array.append(action)
        self.state_array.append(self.state)
        self.state_time_array.append(self.time)
        self.action_time_array.append(self.time)

        lead_state, lead_action = self.lead_car_model.step(self.time, self.state)
        self.lead_state_array.append(lead_state)
        self.lead_action_array.append(lead_action)

        # get observation
        observation = Observation.build(
            ego_state=self.state,
            lead_state=lead_state,
            lead_action=lead_action,
            desired_speed=self.params.desired_speed,
            desired_station=self.params.desired_station,
            lead_car_model=self.lead_car_model,
            time=self.time
        )

        # compute reward
        reward = self.reward_class.compute_reward(
            observation, action, self.lead_car_model, self.time
        )
        self.reward_array.append(reward)

        truncated, terminated = False, False
        if self.time >= self.params.max_time:
            truncated = True
        if self.reward_class.is_collision(observation, self.lead_car_model, self.time):
            terminated = True
        self.time += self.params.dt

        return observation.to_np(), float(reward), terminated, truncated, {}

    def reset(
            self,
            ego_init_state: Optional[State] = None,
            lead_car_model: Optional[LeadCarModel] = None,
            desired_speed: Optional[float] = None,
            desired_station: Optional[float] = None,
            scenario_option_probabilities: Optional[ScenarioOptionProbabilities] = None,
            seed=None,
            options=None,
    ):
        super().reset(seed=seed, options=options)
        self.time = 0.0

        if desired_speed is not None:
            self.params.desired_speed = desired_speed
        else:
            self.params.desired_speed = np.random.uniform(0, self.params.speed_limit)

        if scenario_option_probabilities is None:
            scenario_option_probabilities = ScenarioOptionProbabilities(
                # decel=0.50,
                # const_speed_lead=0.2,
                # stationary_lead=0.05,
                # no_lead=0.2,
                # no_lead_from_standstill=0.05,
                no_lead=0.8,
                no_lead_from_standstill=0.2,
            )
            # scenario_option_probabilities = ScenarioOptionProbabilities(no_lead=1.0)

        scenario_choice = scenario_option_probabilities.sample_scenario()

        if ego_init_state is not None:
            self.state = ego_init_state
        else:
            if scenario_choice == "no_lead_from_standstill":
                self.state = State(
                    station=0.0,
                    speed=0.0,
                    acceleration=0.0,
                )
            else:
                self.state = State(
                    station=0.0,
                    speed=np.random.uniform(0.0, self.params.desired_speed + 5.0),
                    acceleration=np.random.uniform(-2.0, 2.0),
                )

        if desired_station is not None:
            self.params.desired_station = desired_station
        else:
            self.params.desired_station = np.random.uniform(self.state.station + self.state.speed * 5.0 + 50,
                                                            (self.state.station + 600))

        if lead_car_model is None:
            lead_init_state = LeadState(
                station=np.random.uniform(
                    self.state.speed * 0.5 + 3.0, self.state.speed * 4.0 + 10.0
                ),
                speed=np.random.uniform(0.0, self.params.desired_speed + 5.0),
            )

            if scenario_choice == "decel":
                accel = np.random.uniform(-10, 2)
                accel_duration = np.random.uniform(
                    0.5, 2.0 * lead_init_state.speed / (np.abs(accel) + 1e-3)
                )
                self.lead_car_model = DecelLeadModel(
                    init_state=lead_init_state,
                    params=LeadCarModel.Params(dt=self.params.dt),
                    decel_params=DecelLeadModel.DecelParams(
                        time_accel_start=np.random.uniform(0, self.params.max_time),
                        accel=accel,
                        accel_duration=accel_duration,
                    ),
                )
            elif (
                    scenario_choice == "no_lead"
                    or scenario_choice == "no_lead_from_standstill"
            ):
                self.lead_car_model = NoLeadModel()
            elif scenario_choice == "const_speed_lead":
                self.lead_car_model = ConstSpeedLeadModel(
                    init_state=lead_init_state,
                    params=LeadCarModel.Params(dt=self.params.dt),
                )
            elif scenario_choice == "stationary_lead":
                lead_init_state.speed = 0.0
                self.lead_car_model = ConstSpeedLeadModel(
                    init_state=lead_init_state,
                    params=LeadCarModel.Params(dt=self.params.dt),
                )
            else:
                raise ValueError(f"Invalid choice: {scenario_choice}")
        else:
            self.lead_car_model = lead_car_model

        self.state_array = [self.state]
        self.state_time_array = [self.time]
        self.action_time_array = []
        self.action_array = []
        self.reward_array = []

        self.lead_state_array = [self.lead_car_model.state]
        self.lead_action_array = []

        observation = Observation.build(
            ego_state=self.state,
            lead_state=self.lead_car_model.state,
            lead_action=LeadAction(0.0),
            desired_speed=self.params.desired_speed,
            desired_station=self.params.desired_station,
            lead_car_model=self.lead_car_model,
            time=self.time,
        ).to_np()

        if self.render_mode == RenderMode.Human or self.render_mode == RenderMode.Save:
            plt.close()
            self._init_fig()

        info = {}
        return observation, info

    def render(self, file_name=None, title=None):
        if title:
            plt.suptitle(title)
        if self.render_mode == RenderMode.Save:
            self._render()
            plt.savefig(f"{file_name}.png")
        elif self.render_mode == RenderMode.Human:
            self._render()
            plt.pause(self.params.dt / 5)

    def close(self):
        pass


if __name__ == "__main__":
    env = ACCEnv(render_mode=RenderMode.Human)
    check_env(env, warn=True)

    lead_car_model = DecelLeadModel(
        LeadCarModel.Params(dt=env.params.dt),
        init_state=LeadState(station=15.0, speed=0.0),
        decel_params=DecelLeadModel.DecelParams(
            time_accel_start=0.0, accel=0.0, accel_duration=np.inf
        ),
    )
    # lead_car_model = NoLeadModel()
    obs, _ = env.reset(
        ego_init_state=State(station=0, speed=0.0, acceleration=0.0),
        lead_car_model=lead_car_model,
    )

    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())

    done = False
    while done is False and plt.get_fignums():
        obs, reward, terminated, truncated, info = env.step([0.01])
        done = terminated or truncated
        env.render()
    plt.show()
