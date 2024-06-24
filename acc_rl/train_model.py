from acc_env import ACCEnv, RenderMode, ScenarioOptionProbabilities
from eval_policy import eval_policy

from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import os
import numpy as np

exp_name = "PPO_acc_run37_no_lead_accel_lead_bool"

base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "models", exp_name)
image_dir = os.path.join(base_dir, "image", exp_name)
logdir = os.path.join(base_dir, "logs")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Instantiate the env
vec_env = make_vec_env(
    ACCEnv,
    n_envs=1,
    env_kwargs={
        "render_mode": RenderMode.Null,
    },
)
# vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

device = torch.device("cpu")

# Train the agent
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    device=device,
    tensorboard_log=logdir,
    n_steps=2048 * 16,
    batch_size=64 * 4,
    # policy_kwargs=dict(
    #     net_arch=[dict(pi=[32, 32], vf=[32, 32])]  # Adjusted network architecture
    # ),
)

TIMESTEPS = 100000
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
    image_iter_dir = os.path.join(image_dir, f"{TIMESTEPS*iters}")
    os.makedirs(image_iter_dir, exist_ok=True)
    params = ACCEnv.Params()
    # params.max_time = 20
    env = ACCEnv(render_mode=RenderMode.Save, params=params)
    eval_policy(model, env, RenderMode.Save, image_iter_dir, 0)


mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1000)
print(f"Eval Results: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
