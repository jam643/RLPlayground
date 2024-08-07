from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, A2C, DQN, SAC
from lc_env import LcEnv
from simple_road import DubinsRoadGenerator, FixedScenarioRoadGenerator, FixedScenario
from matplotlib import pyplot as plt
import torch
import numpy as np
from matplotlib import cm
from scipy.linalg import solve_continuous_are
import os


#mean_reward, std_reward = evaluate_policy(
#    model,
#    model.get_env(),
#    n_eval_episodes=100,
#)

class EvaluationSet:
    def __init__(self, num_random_roads = 10):
        straight_road_generator = FixedScenarioRoadGenerator(FixedScenario.StraightRoad)
        self.straight_road = straight_road_generator.get_road(seed=0)
        straight_road_generator = FixedScenarioRoadGenerator(FixedScenario.TurnStraightAndWideTurn)
        self.s_curve = straight_road_generator.get_road(seed=0)
        generator_params = DubinsRoadGenerator.DubinsRoadParams(check_intersection=True, num_way_points=4, position_range=200)
        self.dubins_generator = DubinsRoadGenerator(params=generator_params)
        self.random_roads = []
        for i in range(num_random_roads):
            self.random_roads.append(self.dubins_generator.get_road(i))

        self.env = LcEnv(render_mode="save")
        self.max_speed = 0.0

    def get_max_speed(self):
        return self.max_speed

    def get_evaluation_images(self, policy_model, save_dir):
        print("creating evaluation images")
        # straight road
        obs, _ = self.env.reset(road=self.straight_road)
        done = False
        while not done and plt.get_fignums():
            action, _ = policy_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
        speeds = [state.velocity for state in self.env.state_array]
        self.max_speed = max(speeds)
        self.env.render(save_dir=save_dir, file_name="straight_road", title="straight road")
        # S road
        obs, _ = self.env.reset(road=self.s_curve)
        done = False
        while not done and plt.get_fignums():
            action, _ = policy_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
        self.env.render(save_dir=save_dir, file_name="s_road", title="S road")
        # random roads
        for i in range(len(self.random_roads)):
            obs, _ = self.env.reset(road=self.random_roads[i])
            done = False
            while not done and plt.get_fignums():
                action, _ = policy_model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

            self.env.render(save_dir=save_dir, file_name="random_road_" + str(i), title="Random Road" + str(i))

# Random envs
if __name__ == "__main__":
    model_name = "PPO_combined"
    model_steps = "1500000"

    params_for_visualization = LcEnv.Params(start_at_beginning=True, max_time=100)
    env_kwargs = {'params': params_for_visualization}
    vec_env = make_vec_env(LcEnv, env_kwargs=env_kwargs, n_envs=1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model_checkpoints", model_name, model_steps)
    model = PPO.load(os.path.join(base_dir, "model_checkpoints", model_name, model_steps), env=vec_env)

    env = LcEnv(render_mode="human", params=params_for_visualization)
    while plt.get_fignums():
        path_gen_params = DubinsRoadGenerator.DubinsRoadParams(check_intersection=True, num_way_points=4,
                                                               position_range=300)
        path_generator = DubinsRoadGenerator(params=path_gen_params)

        fixed_scenario_generator = FixedScenarioRoadGenerator(scenario=FixedScenario.TurnStraightAndWideTurn)
        obs, _ = env.reset()
        done = False
        while not done and plt.get_fignums():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            env.render(title="Random Env")
            done = terminated or truncated
