from acc_env import ACCEnv, RenderMode, ScenarioOptionProbabilities
from eval_policy import eval_policy
import goal_cost_module
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import os
import numpy as np
exp_name = "PPO_acc_run73"

base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "model_checkpoints", exp_name)
image_dir = os.path.join(base_dir, "image", exp_name)
logdir = os.path.join(base_dir, "logs")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)



# Instantiate the env
# vec_env = make_vec_env(
#     ACCEnv,
#     n_envs=1,
#     env_kwargs={
#         "render_mode": RenderMode.Null,
#     },
# )
# vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

accenv = ACCEnv()
device = torch.device("cpu")

accenv.no_lead_probability = 0.8
accenv.no_lead_from_standstill_probability = 0.2

# Train the agent
model = PPO(
    "MlpPolicy",
    accenv,
    verbose=1,
    device=device,
    tensorboard_log=logdir,
    n_steps=2048 * 1024,
    batch_size=64 * 512,
    policy_kwargs=dict(
        net_arch=[dict(pi=[64, 64], vf=[64, 64])]  # Adjusted network architecture
    ),
)
print(model.policy)
TIMESTEPS = 200000
iters = 0
# while iters < 30:
#     iters += 1
#     model.learn(
#         total_timesteps=TIMESTEPS,
#         reset_num_timesteps=False,
#         tb_log_name=exp_name,
#         progress_bar=True,
#     )
#     model.save(f"{models_dir}/{TIMESTEPS*iters}")
#     image_iter_dir = os.path.join(image_dir, f"{TIMESTEPS*iters}")
#     os.makedirs(image_iter_dir, exist_ok=True)
#     params = ACCEnv.Params()
#     # params.goal_cost = iters * .005
#     # params.max_time = 20
#     env = ACCEnv(render_mode=RenderMode.Save, params=params)
#     env.no_lead_probability = 0.8
#     env.no_lead_from_standstill_probability = 0.2
#     result = eval_policy(model, env, RenderMode.Save, image_iter_dir, 0)

accenv.decel_probability = 0.3
accenv.const_speed_lead_probability = 0.2
accenv.stationary_lead_probability = 0.1
accenv.no_lead_probability = 0.2
accenv.no_lead_from_standstill_probability = 0.2

while iters < 6000:
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
    # params.goal_cost = iters * .005
    # params.max_time = 20
    env = ACCEnv(render_mode=RenderMode.Save, params=params)
    env.decel_probability = 0.3
    env.const_speed_lead_probability = 0.2
    env.stationary_lead_probability = 0.1
    env.no_lead_probability = 0.2
    env.no_lead_from_standstill_probability = 0.2
    result = eval_policy(model, env, RenderMode.Save, image_iter_dir, 0)


mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1000)
print(f"Eval Results: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
