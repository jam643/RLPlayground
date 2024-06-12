import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass
from matplotlib import pyplot as plt
from matplotlib import animation

from stable_baselines3.common.env_checker import check_env


@dataclass
class State:
    station: float
    speed: float


@dataclass
class Action:
    acceleration: float


def motion_model(state: State, action: Action, dt: float) -> State:
    speed = state.speed + action.acceleration * dt
    station = state.station + speed * dt
    return State(station=station, speed=speed)


class DoubleIntegratorEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human", "none", "save"]}

    def __init__(self, render_mode="none", save_dir=None):
        super().__init__()
        self.render_mode = render_mode
        self.save_dir = save_dir

        self.action_space = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )

        self.dt = 0.5
        self.time = 0.0
        self.max_time = 30.0
        self.desired_speed = 0.0
        self.desired_station = 0.0

        if self.render_mode == "human":
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
            ax.set_xlim(0, self.max_time)
            (line,) = ax.plot([], [], ".-")
            self.lines.append(line)
            ax.grid()
            ax.set_xlabel("Time [s]")

        self.axs[self.station_plot_idx].set_ylabel("Station [m]")
        self.axs[self.speed_plot_idx].set_ylabel("Speed [m/s]")
        self.axs[self.acceration_plot_idx].set_ylabel("Acceleration [m/s^2]")
        self.axs[self.reward_plot_idx].set_ylabel("Reward")

    def _update(self, frame):
        if frame == len(self.time_array):
            plt.close()
            self._init_fig()
            return self.lines
        self.lines[self.speed_plot_idx].set_data(
            self.time_array[:frame], [s.speed for s in self.state_array[:frame]]
        )
        self.lines[self.station_plot_idx].set_data(
            self.time_array[:frame], [s.station for s in self.state_array[:frame]]
        )
        self.lines[self.acceration_plot_idx].set_data(
            self.time_array[: max(0, frame - 1)],
            [a.acceleration for a in self.action_array[: max(0, frame - 1)]],
        )
        self.lines[self.reward_plot_idx].set_data(
            self.time_array[: max(0, frame - 1)], self.reward_array[: max(0, frame - 1)]
        )

        self.axs[self.reward_plot_idx].legend(
            [f"Total Reward: {np.sum(self.reward_array):.3f}"],
            loc="center right",
            bbox_to_anchor=(1.0, 0.5),
        )

        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()

        return self.lines

    def step(self, action):
        action = Action(acceleration=action[0])
        self.action_array.append(action)

        truncated, terminated = False, False
        if self.time >= self.max_time:
            truncated = True
            self.render()
        self.time += self.dt
        self.time_array.append(self.time)

        self.state = motion_model(self.state, action, self.dt)
        observation = np.array([self.state.station, self.state.speed]).astype(
            np.float32
        )

        self.state_array.append(self.state)

        # tracking_rew = -0.01 * (self.state.speed - self.desired_speed) ** 2
        tracking_rew = -0.01 * (self.state.station - self.desired_station) ** 2
        tracking_rew += -0.01 * (self.state.speed - self.desired_speed) ** 2
        comfort_rew = -0.05 * action.acceleration**2
        reward = float(tracking_rew + comfort_rew)
        self.reward_array.append(reward)

        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.time = 0.0
        self.time_array = [self.time]

        self.state = State(
            station=np.random.uniform(-10.0, 10.0), speed=np.random.uniform(-10.0, 10.0)
        )
        self.state_array = [self.state]
        self.action_array = []
        self.reward_array = []

        observation = np.array([self.state.station, self.state.speed]).astype(
            np.float32
        )

        info = {}
        return observation, info

    def render(self):
        if self.render_mode == "human":
            # self._init_fig()
            self.ani = animation.FuncAnimation(
                self.fig,
                self._update,
                frames=len(self.time_array) + 1,
                blit=True,
                interval=self.dt * 1000 / 10,
                repeat=False,
            )
            plt.show()

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
    while truncated is False and terminated is False and plt.get_fignums():
        obs, reward, terminated, truncated, info = env.step([1])
