from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, A2C, DQN, SAC
from acc_env import (
    ACCEnv,
    State,
    LeadState,
    DecelLeadModel,
    CutinLeadModel,
    ConstSpeedLeadModel,
    RenderMode,
    LeadCarModel,
    NoLeadModel,
)
from matplotlib import pyplot as plt
import torch
import numpy as np
from matplotlib import cm
from scipy.linalg import solve_continuous_are
import os


def eval_policy(
    model,
    env: ACCEnv,
    render_mode: RenderMode,
    save_dir_name: str = "",
    num_rand_envs: int = int(1e6),
    desired_speed: float = 15,
):

    # Sweep over decels
    cutin_headway_times = np.linspace(4.0, 6.0, 4)
    for cutin_headway_time in cutin_headway_times:
        obs, _ = env.reset(
            ego_init_state=State(station=0, speed=0.9*desired_speed, acceleration=0),
            lead_car_model=CutinLeadModel(
                LeadCarModel.Params(dt=env.params.dt),
                init_state=LeadState(station=0.0, speed=0.0),
                cutin_params=CutinLeadModel.CutinParams(
                    time_cutin=7, relative_speed=0.0, cutin_distance=cutin_headway_time*desired_speed
                ),
            ),
            desired_speed=desired_speed,
            desired_station=2000,
        )
        done = False
        while not done and plt.get_fignums():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            title = f"Lead cutin headway: {cutin_headway_time} sec"
            if render_mode == RenderMode.Human:
                env.render(title=title)
            done = terminated or truncated
        if render_mode == RenderMode.Save:
            env.render(
                title=title,
                file_name=os.path.join(
                    save_dir_name, "cutin_" + str(cutin_headway_time).replace(".", "_")
                ),
            )

    # Sweep over decels
    decels = np.linspace(0, 9, 4)
    for decel in decels:
        obs, _ = env.reset(
            ego_init_state=State(station=0, speed=desired_speed, acceleration=0),
            lead_car_model=DecelLeadModel(
                LeadCarModel.Params(dt=env.params.dt),
                init_state=LeadState(station=desired_speed * 0.5, speed=desired_speed),
                decel_params=DecelLeadModel.DecelParams(
                    time_accel_start=7, accel=-decel, accel_duration=np.inf
                ),
            ),
            desired_speed=desired_speed,
            desired_station=2000,
        )
        done = False
        while not done and plt.get_fignums():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            title = f"Const Decel: {decel}"
            if render_mode == RenderMode.Human:
                env.render(title=title)
            done = terminated or truncated
        if render_mode == RenderMode.Save:
            env.render(
                title=title,
                file_name=os.path.join(
                    save_dir_name, "decel_" + str(decel).replace(".", "_")
                ),
            )

    # Sweep over const speeds to see following distances
    speeds = np.linspace(5, 15, 3)
    for speed in speeds:
        obs, _ = env.reset(
            ego_init_state=State(station=0, speed=speed, acceleration=0),
            lead_car_model=ConstSpeedLeadModel(
                params=LeadCarModel.Params(dt=env.params.dt),
                init_state=LeadState(station=2 * speed, speed=speed),
            ),
            desired_speed=desired_speed,
            desired_station=2000,
        )
        done = False
        while not done and plt.get_fignums():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            title = f"Steady state car follow: {speed}"
            if render_mode == RenderMode.Human:
                env.render(title=title)
            done = terminated or truncated
        if render_mode == RenderMode.Save:
            env.render(
                title=title,
                file_name=os.path.join(
                    save_dir_name, "speed_" + str(speed).replace(".", "_")
                ),
            )

        # Sweep over const speeds to see following distances
    lead_stations = np.linspace(20, 50, 4)
    for lead_station in lead_stations:
        obs, _ = env.reset(
            ego_init_state=State(station=0, speed=speed, acceleration=0),
            lead_car_model=ConstSpeedLeadModel(
                params=LeadCarModel.Params(dt=env.params.dt),
                init_state=LeadState(station=lead_station, speed=0.0),
            ),
            desired_speed=desired_speed,
            desired_station=2000,
        )
        done = False
        while not done and plt.get_fignums():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            title = f"Stationary lead station: {lead_station}"
            if render_mode == RenderMode.Human:
                env.render(title=title)
            done = terminated or truncated
        if render_mode == RenderMode.Save:
            env.render(
                title=title,
                file_name=os.path.join(
                    save_dir_name, "lead_station_" + str(lead_station).replace(".", "_")
                ),
            )

    # Accel from init speed on open road
    init_speeds = np.linspace(0, 15, 4)
    for init_speed in init_speeds:
        obs, _ = env.reset(
            ego_init_state=State(station=0, speed=init_speed, acceleration=0),
            lead_car_model=NoLeadModel(),
            desired_speed=desired_speed,
            desired_station=2000,
        )
        done = False
        while not done and plt.get_fignums():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            title = f"Accel from init speed: {init_speed}"
            if render_mode == RenderMode.Human:
                env.render(title=title)
            done = terminated or truncated
        if render_mode == RenderMode.Save:
            env.render(
                title=title,
                file_name=os.path.join(
                    save_dir_name, "init_speed_" + str(init_speed).replace(".", "_")
                ),
            )

    # Come to stop for st constraint
    stations = np.linspace(75, 150, 4)
    for station in stations:
        obs, _ = env.reset(
            ego_init_state=State(station=0, speed=10.0, acceleration=0),
            lead_car_model=NoLeadModel(),
            desired_speed=desired_speed,
            desired_station=station,
        )
        done = False
        while not done and plt.get_fignums():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            title = f"Stop for station: {station}"
            if render_mode == RenderMode.Human:
                env.render(title=title)
            done = terminated or truncated
        if render_mode == RenderMode.Save:
            env.render(
                title=title,
                file_name=os.path.join(
                    save_dir_name, "stop_station_" + str(station).replace(".", "_")
                ),
            )

    # Random envs
    for i in range(num_rand_envs):
        obs, _ = env.reset()
        done = False
        while not done and plt.get_fignums():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            title = f"Random Env"
            if render_mode == RenderMode.Human:
                env.render(title=title)
            done = terminated or truncated
        if render_mode == RenderMode.Save:
            env.render(
                title=title,
                file_name=os.path.join(save_dir_name, f"random_env_{i}"),
            )


if __name__ == "__main__":
    model_name = "PPO_ACC_V2"
    model_steps = "900000"

    vec_env = make_vec_env(ACCEnv, n_envs=1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model_checkpoints", model_name, model_steps)
    model = PPO.load(
        os.path.join(base_dir, "model_checkpoints", model_name, model_steps), env=vec_env
    )

    mean_reward, std_reward = evaluate_policy(
        model,
        model.get_env(),
        n_eval_episodes=100,
    )

    params = ACCEnv.Params()
    params.max_time = 30
    env = ACCEnv(render_mode=RenderMode.Human, params=params)

    eval_policy(model, env, render_mode=RenderMode.Human, desired_speed=15)
