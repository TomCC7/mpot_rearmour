#!/usr/bin/env python3
"""Rearmour field, capsulation of rearmour as a pure SDF

author: C.C
date: 2023-11-07
"""
from torch_robotics.environments.primitives import PrimitiveShapeField

# from zonopy import batchZonotope # TODO: are they different?
from submodules.rearmour.reachability.conSet import batchZonotope
from submodules.rearmour.distance_net.distance_and_gradient_net import (
    DistanceGradientNet,
)
from submodules.rearmour.distance_net.compute_vertices_from_generators import (
    compute_edges_from_generators,
)
import torch


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

    def render(self, ax, pos=None, ori=None, color=None, **kwargs):
        raise NotImplementedError


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
        self.distance_net = DistanceGradientNet()
        self.distance_net.to(self.tensor_args["device"])

        # hyperplanes and vertices
        combs = self._generate_combinations_upto(max_combs)
        hyperplanes_A, hyperplanes_b = self.obs.polytope(combs)
        hyperplanes_b = hyperplanes_b.unsqueeze(-1)
        self.hyperplanes_A: torch.Tensor = hyperplanes_A.to(self.tensor_args["device"])
        self.hyperplanes_b: torch.Tensor = hyperplanes_b.to(self.tensor_args["device"])
        self.v1, self.v2 = compute_edges_from_generators(
            self.obs.Z[..., 0:1, :],
            self.obs.Z[..., 1:, :],
            self.hyperplanes_A,
            self.hyperplanes_b,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the signed distance

        Args:
            x (torch.Tensor): input points of shape (batch_shape, self.dim)

        Returns:
            signed_distance (torch.Tensor)[n, dim]:
                    signed distance between inputs and nearest obstacle
        """
        assert self.dim == x.shape[-1]
        # flatten
        shape_orig = x.shape[:-1]
        x = x.view(-1, self.dim)

        # transpose to (num_obs * num_points, ...)
        num_points = x.shape[0]
        x = x.repeat(self.num_obs, 1)
        hyperplanes_A = self.hyperplanes_A.repeat_interleave(num_points, 0)
        hyperplanes_b = self.hyperplanes_b.repeat_interleave(num_points, 0)
        v1 = self.v1.repeat_interleave(num_points, 0)
        v2 = self.v2.repeat_interleave(num_points, 0)
        distances, _ = self.distance_net(x, hyperplanes_A, hyperplanes_b, v1, v2)

        return (
            distances.reshape(self.num_obs, num_points)
            .min(dim=0)
            .values.view(*shape_orig)
        )

    def _generate_combinations_upto(self, max_combs):
        return [
            torch.combinations(torch.arange(i, device=self.tensor_args["device"]), 2)
            for i in range(max_combs + 1)
        ]
