from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, A2C, DQN, SAC
from path_env import PathEnv, DubinsPathGenerator
from matplotlib import pyplot as plt
import torch
import numpy as np
from matplotlib import cm
from scipy.linalg import solve_continuous_are
import os

model_name = "PPO_path"
model_steps = "350000"

params_for_visualization = PathEnv.Params(reuse_path=False, start_at_beginning=True)
env_kwargs = {'params': params_for_visualization}
vec_env = make_vec_env(PathEnv, env_kwargs=env_kwargs, n_envs=1)

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", model_name, model_steps)
model = PPO.load(os.path.join(base_dir, "models", model_name, model_steps), env=vec_env)

#mean_reward, std_reward = evaluate_policy(
#    model,
#    model.get_env(),
#    n_eval_episodes=100,
#)

# Random envs

env = PathEnv(render_mode="human", params=params_for_visualization)
while plt.get_fignums():
    path_gen_params = DubinsPathGenerator.DubinsPathParams(check_intersection=True, num_way_points=4)
    path_generator = DubinsPathGenerator(params=path_gen_params)
    obs, _ = env.reset(path_generator=path_generator)
    done = False
    while not done and plt.get_fignums():
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        env.render(title="Random Env")
        done = terminated or truncated
