#!/usr/bin/env python3

from submodules.rearmour.reachability.conSet import batchZonotope
from mpot_rearmour.rearmour_field import RearmourField
import torch

if __name__ == "__main__":
    zono = torch.tensor(
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[3, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ],
        dtype=torch.float,
    )
    field = RearmourField(zono)
    print(
        field.compute_signed_distance(
            torch.tensor([[0, 0, 0], [-5, 0, 0], [10, 0, 0]], dtype=torch.float)
        )
    )
