#!/usr/bin/env python3
from torch_robotics.environments.primitives import PrimitiveShapeField

# from zonopy import batchZonotope # TODO: are they different?
from submodules.rearmour.reachability.conSet import batchZonotope
import torch


class RearmourField(PrimitiveShapeField, tensor_args=None):
    def __init__(self, obs: batchZonotope):
        self.obs: batchZonotope = obs
        self.dim = self.obs.dimension
        self.rearmour_sdf = RearmourDistanceNet(obs, tensor_args=tensor_args)

        super().__init__(self.dim, tensor_args=tensor_args)

    def compute_signed_distance_impl(self, x) -> float:
        """Compute the signed distance

        Args:
            x (torch.Tensor): input points of shape (batch_shape, self.dim)

        Returns:
            signed_distance (float): signed distance between input and nearest obstacle
        """
        return self.rearmour_sdf(x)


class RearmourDistanceNet:
    """Pure functor for rearmour distance net"""

    def __init__(self, obs: batchZonotope, max_combs: int = 200, tensor_args=None):
        """init

        Args:
            obs (batchZonotope): obstacles in the sdf
        """
        self.obs: batchZonotope = obs.to(**tensor_args)
        self.dim = self.obs.dimension
        self.max_combs = max_combs
        self.combs = self._generate_combinations_upto(max_combs)
        self.tensor_args = tensor_args
        self.distance_net = DistanceGradientNet(**tensor_args)
        hyperplanes_A, hyperplanes_b = self.obs.polytope(self.combs)
        hyperplanes_b = hyperplanes_b.unsqueeze(-1)
        self.hyperplanes_A = hyperplanes_A.to(**self.tensor_args)
        self.hyperplanes_b = hyperplanes_b.to(**self.tensor_args)
        self.v1, self.v2 = compute_edges_from_generators(
            self.obs[:, 0:1, :],
            self.obs[:, 1:, :],
            self.hyperplanes_A,
            self.hyperplanes_b,
        )

    def __call__(self, x: torch.Tensor):
        """Compute the signed distance

        Args:
            x (torch.Tensor): input points of shape (batch_shape, self.dim)

        Returns:
            signed_distance (float): signed distance between input and nearest obstacle
        """
        assert self.dim == x.shape[-1]
        x = x.view(-1, self.dim)
        distances, _ = self.distance_net(
            x, self.hyperplanes_A, self.hyperplanes_b, self.v1, self.v2
        )
        return distances.min()

    def _generate_combinations_upto(self, max_combs):
        return [
            torch.combinations(torch.arange(i, device=self.device), 2)
            for i in range(max_combs + 1)
        ]
