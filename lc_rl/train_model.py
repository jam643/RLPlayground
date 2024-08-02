from lc_env import LcEnv, RenderMode
#from eval_policy import eval_policy

from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import os
import numpy as np
from eval_policy import EvaluationSet
from stable_baselines3.common.callbacks import BaseCallback

exp_name = "PPO_combined"

base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "model_checkpoints", exp_name)
image_dir = os.path.join(base_dir, "image", exp_name)
logdir = os.path.join(base_dir, "logs")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Instantiate the env
vec_env = make_vec_env(
    LcEnv,
    n_envs=4,
    env_kwargs={
        "render_mode": RenderMode.Null,
    },
)
# vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

device = torch.device("cpu")

policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]))

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def set_additional_log_values(self, max_speed,e):
        pass
    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        self.logger.record("max_speed", value)
        return True


# Train the agent
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    device=device,
    tensorboard_log=logdir,
    n_steps=2048 * 16,
    batch_size=64 * 4,
    policy_kwargs=policy_kwargs  # Adjusted network architecture
)

eval_set = EvaluationSet()

TIMESTEPS = 100000
iters = 0
while iters < 60:
    iters += 1
    print("iter")
    print(iters)
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name=exp_name,
        progress_bar=True,
    )
    model.save(f"{models_dir}/iteration_{iters}")
    image_iter_dir = os.path.join(image_dir, f"iteration_{iters}")
    os.makedirs(image_iter_dir, exist_ok=True)
    eval_set.get_evaluation_images(model, image_iter_dir, )


#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1000)
#print(f"Eval Results: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
