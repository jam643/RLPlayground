import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass
from matplotlib import pyplot as plt
import time
from typing import Optional
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
from scipy import integrate

from stable_baselines3.common.env_checker import check_env


@dataclass
class State:
    station: float
    speed: float


@dataclass
class Action:
    acceleration: float


@dataclass
class RewWeights:
    station: float = 0.01
    speed: float = 0.01
    acceleration: float = 0.05


class DoubleIntLQR:
    def __init__(self, w: RewWeights, dt: float):
        self.w = w
        self.dt = dt
        self.P, self.K = self.get_disc_lqr_cost2go_and_k_matrix(self.w, self.dt)

    @staticmethod
    def get_disc_lqr_cost2go_and_k_matrix(w: RewWeights, dt: float):
        A = np.array([[0, 1], [0, 0]])
        B = np.array([[0], [1]])
        C = np.array([[1, 0], [0, 1]])  # for full state feedback
        D = np.array([[0], [0]])
        Q = np.diag([w.station, w.speed])
        R = np.array([[w.acceleration]])

        A_d, B_d, _, _, _ = cont2discrete((A, B, C, D), dt)

        P = solve_discrete_are(A_d, B_d, Q, R)

        K = np.linalg.inv(B_d.T @ P @ B_d + R) @ (B_d.T @ P @ A_d)
        K = K[0]
        return P, K

    def get_cost2go(self, state: State):
        return (
            np.array([state.station, state.speed])
            @ self.P
            @ np.array([state.station, state.speed]).T
        )

    def get_command(self, state: State):
        return Action(
            acceleration=(-self.K @ np.vstack([state.station, state.speed]))[0]
        )


def motion_model(state: State, action: Action, dt: float) -> State:
    speed = state.speed + action.acceleration * dt
    station = state.station + speed * dt
    return State(station=station, speed=speed)


def motion_model_rk4(state: State, action: Action, dt: float) -> State:
    def rhs(state, accel):
        speed = state[1]
        return [speed, accel]

    state_init = [state.station, state.speed]
    new_state = integrate.solve_ivp(
        lambda _, state: rhs(state, action.acceleration),
        [0, dt],
        state_init,
        method="RK45",
    ).y[:, -1]
    return State(station=new_state[0], speed=new_state[1])


class DoubleIntegratorEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human", "none", "save"]}

    @dataclass
    class Params:
        dt: float = 0.5
        max_time: float = 30.0
        desired_speed: float = 0.0
        desired_station: float = 0.0
        max_station: float = 10.0
        max_speed: float = 10.0
        max_acceleration: float = 10.0
        reward_weights: RewWeights = RewWeights()

    def __init__(self, render_mode="none", save_dir=None, params=Params()):
        super().__init__()
        self.render_mode = render_mode
        self.save_dir = save_dir
        self.params = params

        self.action_space = spaces.Box(
            low=-self.params.max_acceleration,
            high=self.params.max_acceleration,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        self.time = 0.0

        self.lqr = DoubleIntLQR(self.params.reward_weights, self.params.dt)

        if self.render_mode == "human" or self.render_mode == "save":
            self._init_fig()

    def _init_fig(self):
        self.station_plot_idx = 0
        self.speed_plot_idx = 1
        self.acceration_plot_idx = 2
        self.reward_plot_idx = 3

        self.fig, self.axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        self.axs = self.axs.flatten()
        self.lines = []
        for ax in self.axs:
            ax.set_xlim(0, self.params.max_time)
            (line,) = ax.plot([], [], ".-")
            (line_lqr,) = ax.plot([], [], ".-")
            self.lines.append({"env": line, "lqr": line_lqr})
            ax.grid()
            ax.set_xlabel("Time [s]")

        self.axs[self.station_plot_idx].set_ylabel("Station [m]")
        self.axs[self.speed_plot_idx].set_ylabel("Speed [m/s]")
        self.axs[self.acceration_plot_idx].set_ylabel("Acceleration [m/s^2]")
        self.axs[self.reward_plot_idx].set_ylabel("Reward")

    def _render(self):
        self.lines[self.speed_plot_idx]["env"].set_data(
            self.state_time_array, [s.speed for s in self.state_array]
        )
        self.lines[self.station_plot_idx]["env"].set_data(
            self.state_time_array, [s.station for s in self.state_array]
        )
        self.lines[self.acceration_plot_idx]["env"].set_data(
            self.action_time_array, [a.acceleration for a in self.action_array]
        )
        self.lines[self.reward_plot_idx]["env"].set_data(
            self.action_time_array, self.reward_array
        )

        self.lines[self.speed_plot_idx]["lqr"].set_data(
            self.state_time_array, [s.speed for s in self.lqr_state_array]
        )
        self.lines[self.station_plot_idx]["lqr"].set_data(
            self.state_time_array, [s.station for s in self.lqr_state_array]
        )
        self.lines[self.acceration_plot_idx]["lqr"].set_data(
            self.action_time_array, [a.acceleration for a in self.lqr_action_array]
        )
        self.lines[self.reward_plot_idx]["lqr"].set_data(
            self.action_time_array, self.lqr_reward_array
        )

        self.axs[self.reward_plot_idx].legend(
            [
                f"Reward Per Episode: {np.nansum(self.reward_array)/self.number_episodes:.3f}",
                f"LQR Reward Per Episode: {np.nansum(self.lqr_reward_array)/self.number_episodes:.3f}",
            ],
            loc="center right",
            bbox_to_anchor=(1.0, 0.5),
        )

        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()

        plt.draw()

    def _compute_reward(self, state: State, action: Action):
        tracking_rew = (
            -self.params.reward_weights.station
            * (state.station - self.params.desired_station) ** 2
        )
        tracking_rew += (
            -self.params.reward_weights.speed
            * (state.speed - self.params.desired_speed) ** 2
        )
        comfort_rew = -self.params.reward_weights.acceleration * action.acceleration**2
        return float(tracking_rew + comfort_rew)

    def step(self, action):
        action = Action(acceleration=action[0])
        self.action_array.append(action)
        self.state = motion_model_rk4(self.state, action, self.params.dt)
        observation = np.array([self.state.station, self.state.speed]).astype(
            np.float32
        )
        self.state_array.append(self.state)
        reward = self._compute_reward(self.state, action)
        self.reward_array.append(reward)

        truncated, terminated = False, False
        if self.time >= self.params.max_time:
            truncated = True
        self.time += self.params.dt
        self.state_time_array.append(self.time)
        self.action_time_array.append(self.time)

        # compute optimal LQR actions for comparison
        self.lqr_action_array.append(self.lqr.get_command(self.lqr_state_array[-1]))
        self.lqr_state_array.append(
            motion_model(
                self.lqr_state_array[-1], self.lqr_action_array[-1], self.params.dt
            )
        )
        self.lqr_reward_array.append(
            self._compute_reward(self.lqr_state_array[-1], self.lqr_action_array[-1])
        )

        return observation, reward, terminated, truncated, {}

    def reset(
        self,
        init_state: Optional[State] = None,
        reset_data=True,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed, options=options)
        self.time = 0.0

        if init_state is not None:
            self.state = init_state
        else:
            self.state = State(
                station=np.random.uniform(
                    -self.params.max_station, self.params.max_station
                ),
                speed=np.random.uniform(-self.params.max_speed, self.params.max_speed),
            )

        if reset_data or not hasattr(self, "state_array"):
            self.state_array = [self.state]
            self.state_time_array = [self.time]
            self.action_time_array = []
            self.action_array = []
            self.reward_array = []

            self.lqr_state_array = [self.state]
            self.lqr_action_array = []
            self.lqr_reward_array = []

            self.number_episodes = 1
        else:
            self.state_array.extend([State(np.nan, np.nan), self.state])
            self.state_time_array.extend([np.nan, self.time])
            self.action_time_array.append(np.nan)
            self.action_array.append(Action(np.nan))
            self.reward_array.append(np.nan)

            self.lqr_state_array.extend([State(np.nan, np.nan), self.state])
            self.lqr_action_array.append(Action(np.nan))
            self.lqr_reward_array.append(np.nan)

            self.number_episodes += 1

        observation = np.array([self.state.station, self.state.speed]).astype(
            np.float32
        )

        info = {}
        return observation, info

    def render(self, file_name=None):
        if self.render_mode == "save" and self.time >= self.params.max_time:
            self._render()
            self.fig.suptitle(f"Trained timesteps: {file_name}")
            plt.savefig(f"{self.save_dir}/{file_name}.png")
        elif self.render_mode == "human":
            self._render()
            plt.pause(self.params.dt / 10)

    def close(self):
        pass


if __name__ == "__main__":
    env = DoubleIntegratorEnv(render_mode="human")
    check_env(env, warn=True)

    obs, _ = env.reset()

    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())

    truncated, terminated = False, False
    step = 0
    while truncated is False and terminated is False and plt.get_fignums():
        obs, reward, terminated, truncated, info = env.step([1])
        env.render()
    plt.show()
