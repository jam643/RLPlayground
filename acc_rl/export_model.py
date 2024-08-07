import torch as th
from typing import Tuple
from acc_env import ACCEnv
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy

import os

# Update these with model checkpoint info for tracing
#############
model_name = "PPO_ACC_run55"
model_steps = "9600000"
#############

class OnnxableSB3Policy(th.nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # NOTE: Preprocessing is included, but postprocessing
        # (clipping/inscaling actions) is not,
        # If needed, you also need to transpose the images so that they are channel first
        # use deterministic=False if you want to export the stochastic policy
        # policy() returns `actions, values, log_prob` for PPO
        return self.policy(observation, deterministic=True)


base_dir = os.path.dirname(os.path.abspath(__file__))
model = PPO.load(
    os.path.join(base_dir, "model_checkpoints", model_name, model_steps), device="cpu"
)

onnx_policy = OnnxableSB3Policy(model.policy)

observation_size = model.observation_space.shape
dummy_input = th.randn(1, *observation_size)

onnx_traced_model_path = os.path.join("traced_models", f"{model_name}.onnx")
th.onnx.export(
    onnx_policy,
    dummy_input,
    onnx_traced_model_path,
    opset_version=17,
    input_names=["input"],
)

##### Load and test with onnx

import onnx
import onnxruntime as ort
import numpy as np

onnx_model = onnx.load(onnx_traced_model_path)
onnx.checker.check_model(onnx_model)

observation = np.zeros((1, *observation_size)).astype(np.float32)
ort_sess = ort.InferenceSession(onnx_traced_model_path)
actions, values, log_prob = ort_sess.run(None, {"input": observation})

print(actions, values, log_prob)

# Check that the predictions are the same
with th.no_grad():
    print(model.policy(th.as_tensor(observation), deterministic=True))


# See "ONNX export" for imports and OnnxablePolicy

jit_path = os.path.join("traced_models", f"{model_name}.pt")

# Trace and optimize the module
traced_module = th.jit.trace(onnx_policy.eval(), dummy_input)
frozen_module = th.jit.freeze(traced_module)
frozen_module = th.jit.optimize_for_inference(frozen_module)
th.jit.save(frozen_module, jit_path)

##### Load and test with torch

dummy_input = th.randn(1, *observation_size)
loaded_module = th.jit.load(jit_path)
action_jit = loaded_module(dummy_input)
