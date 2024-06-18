import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass, fields
from matplotlib import pyplot as plt
import time
from typing import Optional
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


def motion_model(
    state: State, action: Action, max_accel: float, min_accel: float, dt: float
) -> State:
    acceleration = state.acceleration + action.jerk * dt
    acceleration = np.clip(acceleration, min_accel, max_accel)
    speed = state.speed + acceleration * dt
    if speed < 0:
        speed = 0
        acceleration = 0
    station = state.station + speed * dt
    return State(station=station, speed=speed, acceleration=acceleration)


def motion_model_lead(state: LeadState, action: LeadAction, dt: float) -> LeadState:
    speed = state.speed + action.acceleration * dt
    if speed < 0:
        speed = 0
        action.acceleration = 0
    station = state.station + speed * dt
    return LeadState(station=station, speed=speed)


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

    def step(self, time: float):
        action = self.get_action(time, self.state)
        self.state = motion_model_lead(self.state, action, self.params.dt)
        return self.state, action


@dataclass
class Observation:
    relative_station: float
    lead_speed: float
    relative_accel: float
    ego_speed: float
    ego_acceleration: float
    desired_speed_error: float
    is_lead: float

    @staticmethod
    def build(
        ego_state: State,
        lead_state: LeadState,
        lead_action: LeadAction,
        desired_speed: float,
        lead_car_model: LeadCarModel,
    ):
        relative_station = lead_state.station - ego_state.station
        lead_speed = lead_state.speed
        relative_accel = lead_action.acceleration - ego_state.acceleration
        is_lead = 1.0
        if isinstance(lead_car_model, NoLeadModel):
            relative_station = 0.0
            lead_speed = 0.0
            relative_accel = 0.0
            is_lead = 0.0

        return Observation(
            relative_station=relative_station,
            lead_speed=lead_speed,
            relative_accel=relative_accel,
            ego_speed=ego_state.speed,
            ego_acceleration=ego_state.acceleration,
            desired_speed_error=ego_state.speed - desired_speed,
            is_lead=is_lead,
        )

    def to_np(self):
        return np.array([getattr(self, field.name) for field in fields(self)]).astype(
            np.float32
        )


class Reward:
    @dataclass
    class Params:
        speed_weight: float = 0.05
        accel_weight: float = 0.2
        jerk_weight: float = 0.05
        clearance_weight: float = 10.0
        collision_weight: float = 10.0
        clearance_buffer_m: float = 2.0

    def __init__(self, params: Params):
        self.params = params

    def compute_reward(
        self, obs: Observation, action: Action, lead_car_model: LeadCarModel
    ):
        comfort_accel_cost = self.params.accel_weight * obs.ego_acceleration**2

        comfort_cost = comfort_accel_cost + self.params.jerk_weight * action.jerk**2

        tracking_cost = self.params.speed_weight * np.abs(obs.desired_speed_error)

        safety_cost = 0
        rel_speed = obs.lead_speed - obs.ego_speed
        if self.is_collision(obs, lead_car_model) and rel_speed <= 0:
            safety_cost = (
                self.params.collision_weight * (obs.lead_speed - obs.ego_speed) ** 2
            )

        clearance_cost = 0
        if obs.relative_station <= self.params.clearance_buffer_m and not isinstance(
            lead_car_model, NoLeadModel
        ):
            clearance_cost = (
                self.params.clearance_weight
                * (obs.relative_station - self.params.clearance_buffer_m) ** 2
            )

        return -comfort_cost - tracking_cost - safety_cost - clearance_cost

    def is_collision(self, obs: Observation, lead_car_model: LeadCarModel):
        return obs.relative_station <= 0.0 and not isinstance(
            lead_car_model, NoLeadModel
        )


class ConstSpeedLeadModel(LeadCarModel):
    def get_action(self, time: float, state: LeadState) -> LeadAction:
        return LeadAction(acceleration=0.0)


class NoLeadModel(LeadCarModel):
    def __init__(self):
        super().__init__(params=LeadCarModel.Params(dt=0.0), init_state=LeadState(0, 0))

    def get_action(self, time: float, state: LeadState) -> LeadAction:
        return LeadAction(acceleration=0.0)

    def step(self, time: float):
        return LeadState(0, 0), LeadAction(0.0)


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
            self.decel_params.time_accel_start <= time
            and time
            < self.decel_params.time_accel_start + self.decel_params.accel_duration
        ):
            action = LeadAction(acceleration=self.decel_params.accel)
        return action


class ACCEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    @dataclass
    class Params:
        dt: float = 0.5
        max_time: float = 15.0
        max_jerk: float = 20.0
        max_accel: float = 2.0
        min_accel: float = -7.0
        speed_limit: float = 20.0
        desired_speed: float = speed_limit

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

        self.fig, self.axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
        plt.subplots_adjust(hspace=0.1, wspace=0.1)
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

        self.lines[self.speed_plot_idx]["lead"].set_data(
            self.state_time_array, [s.speed for s in self.lead_state_array]
        )
        self.lines[self.station_plot_idx]["lead"].set_data(
            self.state_time_array, [s.station for s in self.lead_state_array]
        )
        self.lines[self.acceration_plot_idx]["lead"].set_data(
            self.action_time_array, [a.acceleration for a in self.lead_action_array]
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
            -self.params.max_jerk, self.params.max_jerk
        )
        self.axs[self.speed_plot_idx].set_ylim(0.0, self.params.speed_limit + 1.0)
        self.axs[self.headway_time_plot_idx].set_ylim(0.0, 10.0)

        plt.draw()

    def step(self, action):
        # update state
        action = Action(jerk=action[0] * self.params.max_jerk)
        self.action_array.append(action)
        self.state = motion_model(
            self.state,
            action,
            min_accel=self.params.min_accel,
            max_accel=self.params.max_accel,
            dt=self.params.dt,
        )
        self.state_array.append(self.state)
        self.state_time_array.append(self.time)
        self.action_time_array.append(self.time)

        lead_state, lead_action = self.lead_car_model.step(self.time)
        self.lead_state_array.append(lead_state)
        self.lead_action_array.append(lead_action)

        # get observation
        observation = Observation.build(
            ego_state=self.state,
            lead_state=lead_state,
            lead_action=lead_action,
            desired_speed=self.params.desired_speed,
            lead_car_model=self.lead_car_model,
        )

        # compute reward
        reward = self.reward_class.compute_reward(
            observation, action, self.lead_car_model
        )
        self.reward_array.append(reward)

        truncated, terminated = False, False
        if self.time >= self.params.max_time:
            truncated = True
        if self.reward_class.is_collision(observation, self.lead_car_model):
            terminated = True
        self.time += self.params.dt

        return observation.to_np(), float(reward), terminated, truncated, {}

    def reset(
        self,
        ego_init_state: Optional[State] = None,
        lead_car_model: Optional[LeadCarModel] = None,
        desired_speed: Optional[float] = None,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed, options=options)
        self.time = 0.0

        if desired_speed is not None:
            self.params.desired_speed = desired_speed
        else:
            self.params.desired_speed = np.random.uniform(0, self.params.speed_limit)

        if ego_init_state is not None:
            self.state = ego_init_state
        else:
            self.state = State(
                station=0.0,
                speed=np.random.uniform(0.0, self.params.desired_speed + 5.0),
                acceleration=np.random.uniform(-2.0, 2.0),
            )

        if lead_car_model is None:
            lead_init_state = LeadState(
                station=np.random.uniform(
                    self.state.speed * 0.5 + 3.0, self.state.speed * 4.0 + 10.0
                ),
                speed=np.random.uniform(0.0, self.params.desired_speed + 5.0),
            )

            options = {"decel": 0.70, "const_speed": 0.2, "no_lead": 0.1}
            # options = {"decel": 0.0, "const_speed": 0.0, "no_lead": 1.0}
            choice = np.random.choice(list(options.keys()), p=list(options.values()))
            if choice == "decel":
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
            elif choice == "no_lead":
                self.lead_car_model = NoLeadModel()
            else:
                self.lead_car_model = ConstSpeedLeadModel(
                    init_state=lead_init_state,
                    params=LeadCarModel.Params(dt=self.params.dt),
                )
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
            lead_car_model=self.lead_car_model,
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

    obs, _ = env.reset()

    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())

    truncated = False
    while truncated is False and plt.get_fignums():
        obs, reward, terminated, truncated, info = env.step([0.3])
        env.render()
    plt.show()
