#!/usr/bin/env python3
import torch


class RearmourEnv:
    """Env to be used for mpot planning"""
    def __init_(self, name="Rearmour", dim: int = 3):
        self.dim = dim
        self.limits = torch.ones([self.dim, 2])
        self.limits[:, 0] = -1

        self.obj_extra_list: list = None

    def build_occupancy_map(self, *args):
        raise NotImplementedError("rearmour env doesn't support occupancy map")

    def get_df_obj_list(self):
        pass
