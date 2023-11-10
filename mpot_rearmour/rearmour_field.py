#!/usr/bin/env python3
"""Rearmour field, capsulation of rearmour as a pure SDF

author: C.C
date: 2023-11-07
"""
from torch_robotics.environments.primitives import PrimitiveShapeField

from zonopy import batchZonotope
from submodules.rearmour.distance_net.batched_distance_and_gradient_net import (
    BatchedDistanceGradientNet,
)
from submodules.rearmour.distance_net.compute_vertices_from_generators import (
    compute_edges_from_generators,
)
from mpot_rearmour.zono_utils_3d import plot3d
import torch
import numpy as np


class RearmourField(PrimitiveShapeField):
    def __init__(self, obs: batchZonotope | torch.Tensor, tensor_args=None):
        if isinstance(obs, torch.Tensor):
            obs = batchZonotope(obs)
        elif not isinstance(obs, batchZonotope):
            raise TypeError(
                f"obs should be of type batchZonotope or torch.Tensor, got {type(obs)}"
            )

        if tensor_args is None:
            tensor_args = {"device": obs.device, "dtype": obs.dtype}

        self.obs: batchZonotope = obs
        self.dim = self.obs.dimension
        self.render_colors = np.random.rand(self.obs.batch_shape[0], 3)
        self.rearmour_sdf = RearmourDistanceNet(obs, tensor_args=tensor_args)

        super().__init__(self.dim, tensor_args=tensor_args)

    def compute_signed_distance_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the signed distance

        Args:
            x (torch.Tensor): input points of shape (batch_shape, self.dim)

        Returns:
            signed_distance (torch.Tensor)[n, dim]:
                    signed distance between inputs and nearest obstacle
        """
        return self.rearmour_sdf(x)

    def add_to_occupancy_map(self, obst_map):
        # Adds obstacle to an occupancy grid
        raise NotImplementedError

    def render(self, ax, pos=None, ori=None, color="gray", cmap="gray", **kwargs):
        assert ax.name == "3d"
        for i in range(self.obs.Z.shape[0]):
            obs_single = self.obs[i]
            plot3d(
                obs_single,
                color=self.render_colors[i],
                alpha=0.8,
                vertex=True,
                figax=(None, ax),
            )


class RearmourDistanceNet:
    """Pure functor for rearmour distance net"""

    def __init__(self, obs: batchZonotope, max_combs: int = 200, tensor_args=None):
        """init

        Args:
            obs (batchZonotope): obstacles in the sdf
        """
        self.dim = 3
        assert obs.dimension == self.dim, "only support 3D obstacles"

        self.tensor_args = tensor_args
        if self.tensor_args is None:
            self.tensor_args = {"device": obs.device, "dtype": obs.dtype}

        self.obs: batchZonotope = obs.to(**tensor_args)
        self.num_obs: int = self.obs.batch_shape[0]
        self.distance_net = BatchedDistanceGradientNet()
        self.distance_net.to(self.tensor_args["device"])

        # hyperplanes and vertices
        combs = self._generate_combinations_upto(max_combs)
        self.hyperplanes_A, self.hyperplanes_b = self.obs.polytope(combs)
        self.hyperplanes_b = self.hyperplanes_b.unsqueeze(-1)
        self.v1, self.v2 = compute_edges_from_generators(
            self.obs.Z[..., 0:1, :],
            self.obs.Z[..., 1:, :],
            self.hyperplanes_A,
            self.hyperplanes_b,
        )
        self.hyperplanes_b = self.hyperplanes_b.squeeze(-1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the signed distance

        Args:
            x (torch.Tensor): input points of shape (batch_shape, self.dim)

        Returns:
            signed_distance (torch.Tensor)[n, dim]:
                    signed distance between inputs and nearest obstacle
        """
        assert self.dim == x.shape[-1]
        with torch.no_grad():
            # flatten
            shape_orig = x.shape[:-1]
            x = x.view(-1, self.dim)

            distances, _ = self.distance_net(
                x, self.hyperplanes_A, self.hyperplanes_b, self.v1, self.v2
            )

        return distances.min(dim=1).values.view(*shape_orig)

    def _generate_combinations_upto(self, max_combs):
        return [
            torch.combinations(torch.arange(i, device=self.tensor_args["device"]), 2)
            for i in range(max_combs + 1)
        ]
