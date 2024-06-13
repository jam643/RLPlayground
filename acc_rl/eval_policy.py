from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, A2C, DQN, SAC
from acc_env import ACCEnv, State, LeadState, DecelPolicy, ConstSpeedPolicy
from matplotlib import pyplot as plt
import torch
import numpy as np
from matplotlib import cm
from scipy.linalg import solve_continuous_are
import os

model_name = "PPO_acc"
model_steps = "200000"

vec_env = make_vec_env(ACCEnv, n_envs=1)

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", model_name, model_steps)
model = PPO.load(os.path.join(base_dir, "models", model_name, model_steps), env=vec_env)

mean_reward, std_reward = evaluate_policy(
    model,
    model.get_env(),
    n_eval_episodes=100,
)

# Sweep over decels
env = ACCEnv(render_mode="human")
decels = np.linspace(0, 9, 4)
for decel in decels:
    obs, _ = env.reset(
        ego_init_state=State(station=0, speed=15, acceleration=0),
        lead_init_state=LeadState(station=20, speed=15),
        lead_policy=DecelPolicy(
            DecelPolicy.Params(time_accel_start=5, accel=-decel, accel_duration=np.inf)
        ),
    )
    done = False
    while not done and plt.get_fignums():
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        env.render(title=f"Const Decel: {decel}")
        done = terminated or truncated

# Sweep over const speeds to see following distances
env = ACCEnv(render_mode="human")
speeds = np.linspace(5, 15, 3)
for speed in speeds:
    obs, _ = env.reset(
        ego_init_state=State(station=0, speed=speed, acceleration=0),
        lead_init_state=LeadState(station=2 * speed, speed=speed),
        lead_policy=ConstSpeedPolicy(),
    )
    done = False
    while not done and plt.get_fignums():
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        env.render(title=f"Const Speed: {speed}")
        done = terminated or truncated

# Random envs
env = ACCEnv(render_mode="human")
while plt.get_fignums():
    obs, _ = env.reset()
    done = False
    while not done and plt.get_fignums():
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        env.render(title="Random Env")
        done = terminated or truncated
