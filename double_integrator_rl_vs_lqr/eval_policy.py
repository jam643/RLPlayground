from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, A2C, DQN, SAC
from double_integrator_env import DoubleIntegratorEnv
from matplotlib import pyplot as plt
import torch
import numpy as np
from matplotlib import cm
from scipy.linalg import solve_continuous_are
import os

model_name = "PPO_double_integrator"
model_steps = "150000"

vec_env = make_vec_env(
    DoubleIntegratorEnv, n_envs=1, env_kwargs={"render_mode": "human"}
)

base_path = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(base_path, f"models/{model_name}/{model_steps}")
model = PPO.load(model_dir, env=vec_env)

mean_reward, std_reward = evaluate_policy(
    model,
    model.get_env(),
    n_eval_episodes=100,
)

obs = vec_env.reset()

done = False
step = 0
while plt.get_fignums():
    action, _ = model.predict(obs, deterministic=False)
    step += 1
    obs, returns, done, info = vec_env.step(action)
    if done:
        vec_env.reset()
    # vec_env.render()
