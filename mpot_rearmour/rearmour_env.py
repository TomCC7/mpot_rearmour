#!/usr/bin/env python3
"""Rearmour environment implementation following torch_robotics

author: C.C
date: 2023-11-07
"""
from torch_robotics.environments.env_base import EnvBase
from mpot_rearmour.rearmour_field import RearmourField
from torch_robotics.environments.primitives import ObjectField
import torch


class RearmourEnv(EnvBase):
    def __init__(
        self,
        obs: torch.Tensor,
        name="rearmour_env",
        limits: float = 1,
        tensor_args=None,
        **kwargs
    ):
        self.tensor_args = (
            tensor_args
            if tensor_args is not None
            else {"device": obs.device, "dtype": obs.dtype}
        )
        rearmour_field = RearmourField(obs, tensor_args=tensor_args)
        obj_field = ObjectField([rearmour_field], "rearmour_zonotopes")
        obj_list = [obj_field]
        limits = torch.ones((2, 3), **self.tensor_args) * limits
        limits[0, :] *= -1
        super().__init__(
            name=name,
            limits=limits,  # environments limits
            obj_fixed_list=obj_list,
            tensor_args=tensor_args,
            **kwargs
        )
