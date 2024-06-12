from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, A2C, DQN, SAC
from double_integrator_rl_vs_lqr.double_integrator_env import DoubleIntegratorEnv
from matplotlib import pyplot as plt
import torch
import numpy as np
from matplotlib import cm
from scipy.linalg import solve_continuous_are
import os

model_name = "PPO_1"
model_steps = "150000"

vec_env = make_vec_env(DoubleIntegratorEnv, n_envs=1)

base_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(base_path, f"models/{model_name}/{model_steps}")

model = PPO.load(model_path, env=vec_env)


station = np.linspace(-10, 10, 101)
speed = np.linspace(-10, 10, 101)
station, speed = np.meshgrid(station, speed)
ppo_value_fn = np.zeros_like(station)
for i in range(101):
    for j in range(101):
        ppo_value_fn[i, j] = (
            model.policy.value_net(
                model.policy.mlp_extractor.value_net(
                    torch.Tensor([station[i, j], speed[i, j]])
                )
            )
            .flatten()
            .detach()
            .numpy()[0]
        )
ppo_value_fn = ppo_value_fn - np.max(ppo_value_fn)

fig, axs = plt.subplots(nrows=2, ncols=2, subplot_kw={"projection": "3d"})
axs = axs.flatten()
axs[0].plot_surface(
    station,
    speed,
    ppo_value_fn,
    cmap=cm.coolwarm,
)
axs[0].contourf(
    station, speed, ppo_value_fn, zdir="z", offset=np.min(ppo_value_fn), cmap="coolwarm"
)

axs[0].set_xlabel("Station [m]")
axs[0].set_ylabel("Speed [m/s]")
axs[0].set_zlabel("Reward to go")
axs[0].set_title("PPO Value Function Network")

A = np.array([[0, 1], [0, 0]])
B = np.array([[0], [1]])
Q = np.diag([0.01, 0.01])
R = np.array([[0.05]])

P = solve_continuous_are(A, B, Q, R)
lqr_value_fn = np.zeros_like(station)
for i in range(101):
    for j in range(101):
        lqr_value_fn[i, j] = -(
            np.array([station[i, j], speed[i, j]])
            @ P
            @ np.array([station[i, j], speed[i, j]]).T
        )

axs[1].plot_surface(
    station,
    speed,
    lqr_value_fn,
    cmap=cm.coolwarm,
)
axs[1].contourf(
    station, speed, lqr_value_fn, zdir="z", offset=np.min(lqr_value_fn), cmap="coolwarm"
)

axs[1].set_xlabel("Station [m]")
axs[1].set_ylabel("Speed [m/s]")
axs[1].set_zlabel("Reward to go")
axs[1].set_title("LQR Value Function")

ppo_value_fn_error = np.abs(ppo_value_fn - lqr_value_fn)
axs[2].plot_surface(
    station,
    speed,
    ppo_value_fn_error,
    cmap=cm.coolwarm,
)
axs[2].contourf(
    station,
    speed,
    ppo_value_fn_error,
    zdir="z",
    offset=np.min(ppo_value_fn_error),
    cmap="coolwarm",
)
axs[2].set_xlabel("Station [m]")
axs[2].set_ylabel("Speed [m/s]")
axs[2].set_zlabel("Reward to go")
axs[2].set_title("PPO Value Function Error")

plt.show()
