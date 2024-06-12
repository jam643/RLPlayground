from acc_env import AccEnv

from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from matplotlib import pyplot as plt
import torch
import time
import os

exp_name = f"PPO_1"

models_dir = f"models/{exp_name}/"
logdir = f"logs/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Instantiate the env
vec_env = make_vec_env(AccEnv, n_envs=1, env_kwargs={"render_mode": "human"})
device = torch.device("cpu")

# Train the agent
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    device=device,
    tensorboard_log=logdir,
    gamma=1.0,
)

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

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

# render the policy with random envs
obs = vec_env.reset()
done = False
step = 0
while plt.get_fignums():
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)

    vec_env.render()
