from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, A2C, DQN, SAC
from acc_env import ACCEnv, State, LeadState, DecelPolicy, ConstSpeedPolicy, RenderMode
from matplotlib import pyplot as plt
import torch
import numpy as np
from matplotlib import cm
from scipy.linalg import solve_continuous_are
import os


def eval_policy(
    model, render_mode, save_dir_name: str = "", num_rand_envs: int = np.inf
):

    # Accel from init speed on open road
    init_speeds = np.linspace(0, 15, 4)
    for init_speed in init_speeds:
        env = ACCEnv(render_mode=render_mode)
        obs, _ = env.reset(
            ego_init_state=State(station=0, speed=init_speed, acceleration=0),
            lead_init_state=LeadState(station=-15.0, speed=0),
            lead_policy=ConstSpeedPolicy(),
        )
        done = False
        while not done and plt.get_fignums():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if render_mode == RenderMode.Human:
                env.render(title=f"Accel from init speed: {init_speed}")
            done = terminated or truncated
        if render_mode == RenderMode.Save:
            env.render(
                file_name=os.path.join(
                    save_dir_name, "init_speed_" + str(init_speed).replace(".", "_")
                )
            )

    # Sweep over decels
    env = ACCEnv(render_mode=render_mode)
    decels = np.linspace(0, 9, 4)
    for decel in decels:
        obs, _ = env.reset(
            ego_init_state=State(station=0, speed=15, acceleration=0),
            lead_init_state=LeadState(station=20, speed=15),
            lead_policy=DecelPolicy(
                DecelPolicy.Params(
                    time_accel_start=5, accel=-decel, accel_duration=np.inf
                )
            ),
        )
        done = False
        while not done and plt.get_fignums():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            env.render(title=f"Const Decel: {decel}")
            if render_mode == RenderMode.Human:
                env.render(title=f"Accel from init speed: {init_speed}")
            done = terminated or truncated
        if render_mode == RenderMode.Save:
            env.render(
                file_name=os.path.join(
                    save_dir_name, "decel_" + str(decel).replace(".", "_")
                )
            )

    # Sweep over const speeds to see following distances
    env = ACCEnv(render_mode=render_mode)
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
            if render_mode == RenderMode.Human:
                env.render(title=f"Accel from init speed: {init_speed}")
            done = terminated or truncated
        if render_mode == RenderMode.Save:
            env.render(
                file_name=os.path.join(
                    save_dir_name, "speed_" + str(speed).replace(".", "_")
                )
            )

    # Random envs
    env = ACCEnv(render_mode=render_mode)
    for i in range(num_rand_envs):
        obs, _ = env.reset()
        done = False
        while not done and plt.get_fignums():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            env.render(title="Random Env")
            if render_mode == RenderMode.Human:
                env.render(title=f"Accel from init speed: {init_speed}")
            done = terminated or truncated
        if render_mode == RenderMode.Save:
            env.render(file_name=os.path.join(save_dir_name, f"random_env_{i}"))


if __name__ == "__main__":
    model_name = "PPO_no_lead"
    model_steps = "300000"

    vec_env = make_vec_env(ACCEnv, n_envs=1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", model_name, model_steps)
    model = PPO.load(
        os.path.join(base_dir, "models", model_name, model_steps), env=vec_env
    )

    mean_reward, std_reward = evaluate_policy(
        model,
        model.get_env(),
        n_eval_episodes=100,
    )

    eval_policy(model, render_mode=RenderMode.Human)
