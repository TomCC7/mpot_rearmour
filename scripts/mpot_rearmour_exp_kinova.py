#!/usr/bin/env python3
import os
import time
from pathlib import Path
import torch
import numpy as np

from mpot_rearmour.mpot_rearmour_task import MPOTRearmourTask
from mpot_rearmour.robots import RobotGen3
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device


def run_kinova_scene(scene_file: str, exp_name: str):
    seed = int(time.time())
    fix_random_seed(seed)

    device = get_torch_device()
    tensor_args = {"device": device, "dtype": torch.float32}
    print(f"running experiment: {exp_name}")
    print(f"Using device: {device}")

    # ---------------------------- Read csv file ---------------------------------
    settings = np.loadtxt(scene_file, delimiter=",")
    start_state = torch.from_numpy(settings[0, :]).to(**tensor_args)
    goal_state = torch.from_numpy(settings[1, :]).to(**tensor_args)
    obs_settings = torch.from_numpy(settings[3:, :-1]).to(**tensor_args)
    obs = torch.cat(
        [obs_settings[:, :3].unsqueeze(1), torch.diag_embed(obs_settings[:, 3:]) / 2],
        dim=1,
    )
    print("Start state: ", start_state)
    print("Goal state: ", goal_state)
    print(f"obs: {obs}")

    task = MPOTRearmourTask(
        start_state,
        goal_state,
        obs,
        exp_name,
        RobotGen3,
        tensor_args=tensor_args,
        seed=seed,
    )
    task.run()


if __name__ == "__main__":
    csv_path: str = "./submodules/rearmour/kinova_rdf_scenarios/scene_003.csv"
    exp_name: str = os.path.basename(csv_path).removesuffix(".csv")
    run_kinova_scene(csv_path, exp_name)
