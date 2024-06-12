from double_integrator_env import DoubleIntegratorEnv, State

from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from matplotlib import pyplot as plt
import torch
import time
import os
import numpy as np

exp_name = f"PPO_double_integrator"

base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "models", exp_name)
image_dir = os.path.join(base_dir, "image", exp_name)
logdir = os.path.join(base_dir, "logs")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Instantiate the env
vec_env = make_vec_env(
    DoubleIntegratorEnv,
    n_envs=1,
    env_kwargs={"render_mode": "none"},
)

device = torch.device("cpu")

# Train the agent
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    device=device,
    tensorboard_log=logdir,
    gamma=1.0,  # no discounting to make comparable with LQR
)

init_states = [State(station=s, speed=0.0) for s in np.linspace(-10, 10, 11)]

TIMESTEPS = 10000
iters = 0
while iters < 15:
    iters += 1
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name=exp_name,
        progress_bar=True,
    )
    model.save(f"{models_dir}/{TIMESTEPS*iters}")

    env = DoubleIntegratorEnv(render_mode="save", save_dir=image_dir)
    for state in init_states:
        obs, _ = env.reset(init_state=state, reset_data=False)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.render(file_name=f"{TIMESTEPS*iters}")

    evaluate_policy(model, model.get_env(), n_eval_episodes=1, render=True)


mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
print(f"Eval Results: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
