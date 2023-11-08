#!/usr/bin/env python3
"""Attempts to extend hyperplane calculations to n dimension.

:author: C.C
"""
from zonopy import zonotope
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from typing import Optional


def reduce_box(zono: zonotope, order: float = 1.0) -> zonotope:
    """Reduce zonotope by boxing.

    :param: zono: zonotope
    :param: order: order of reduced zonotope. order = num_generators / dim,
                   >= 1 so that geometry don't collapse
    :return: reduced zonotope
    :ref: https://ieeexplore.ieee.org/document/8264508
    """
    assert order >= 1.0, "order must be >= 1.0"
    gens = zono.generators
    if order > 1:
        num_gens = ceil(order * zono.dimension)
        assert num_gens <= zono.n_generators, "order too large"
        sort_vals = gens.abs().sum(dim=1) - gens.abs().max(dim=1)[0]
        _, sort_idx = torch.sort(sort_vals, descending=False)
        gens_sorted = gens[sort_idx, :]
        num_extra = ceil((order - 1) * zono.dimension)
        gens1 = torch.diag(gens_sorted[:-num_extra, :].abs().sum(dim=0))
        gens2 = gens_sorted[-num_extra:, :]
        new_gens = torch.vstack((gens1, gens2))
    else:
        new_gens = torch.diag(gens.abs().sum(dim=0))

    return zonotope(torch.vstack((zono.center, new_gens)))


def reduce_PCA(zono: zonotope, order: float = 1.0, eps: float = 1e-6) -> zonotope:
    """Reduce zonotope by PCA.

    :param zono: zonotope
    :param order: order of reduced zonotope. order = num_generators / dim
    :param eps: epsilon added to zonotope for numerical stability
    :return: reduced zonotope
    :ref: https://ieeexplore.ieee.org/document/8264508
    """
    assert order >= 1.0, "order must be >= 1.0"
    zono = zonotope(zono.Z + eps)
    gens = torch.vstack((zono.generators, -zono.generators))
    cov = gens.T @ gens
    trans, _, _ = torch.linalg.svd(cov, full_matrices=True)

    zono_trans = zonotope(torch.vstack([zono.center, (trans.T @ zono.generators.T).T]))
    zono_box = reduce_box(zono_trans, order)
    return zonotope(torch.vstack([zono.center, (trans @ zono_box.generators.T).T]))


def vertices(zono: zonotope) -> torch.Tensor:
    """Get vertices of a planar zonotope.

    NOTE: zonotope must be "2D" (i.e. rank of generators matrix is 2)
    TODO: speed up idea: translate to 2D coordinate for faster calculation,
          and translate back

    :param zono: zonotope
    :return: vertices of zonotope
    """
    if not isinstance(zono, zonotope):
        raise TypeError("Input must be a zonotope!")

    dim = torch.linalg.matrix_rank(zono.generators)

    if dim != 2:
        raise RuntimeError("rank of generators matrix must be 2!")

    # preparations
    center = zono.center
    generators = zono.generators

    # calculate norm_vectors
    plane_norm = torch.cross(generators[0, :], generators[1, :])
    norm_vectors = torch.cross(generators, plane_norm.expand_as(generators), dim=1)

    # NOTE: summing all other generators up with coefficient 1 or -1,
    # determined by dot product of the incoming generator and the hyperplane normal
    # (2m, d)
    centers = torch.ones(
        (2 * zono.n_generators, zono.dimension), dtype=zono.dtype
    ) * center.unsqueeze(0)
    for i in range(zono.n_generators):
        coeffs = ((generators @ norm_vectors[[i], :].T) > 0).float()  # n cp_gens x 1
        coeffs[coeffs == 0] = -1
        coeffs[i, :] = 0  # not contributing to center
        center_offset = torch.sum(coeffs * generators, dim=0)
        centers[2 * i, :] += center_offset
        centers[2 * i + 1, :] -= center_offset

    # (2, 2m, d) (vertices, edges, dim)
    edges = torch.repeat_interleave(
        torch.tensor([1, -1], dtype=zono.dtype).reshape(2, 1, 1)
        * generators.unsqueeze(0),
        repeats=2,
        dim=1,
    ) + centers.unsqueeze(0)

    # connect edges and extract vertices
    vtx = torch.zeros((2 * zono.n_generators, zono.dimension), dtype=zono.dtype)
    vtx[0, :] = edges[0, 0, :]
    prev_id = 0
    for i in range(1, 2 * zono.n_generators):
        for j in range(2 * zono.n_generators):
            if j == prev_id:
                continue

            if torch.allclose(edges[0, j, :], vtx[i - 1, :]):
                vtx[i, :] = edges[1, j, :]
                prev_id = j
                break
            elif torch.allclose(edges[1, j, :], vtx[i - 1, :]):
                vtx[i, :] = edges[0, j, :]
                prev_id = j
                break

    return vtx


def decompose(zono: zonotope) -> list[zonotope]:
    """Return all hyperplanes of a zonotope.

    This can be viewed as a decomposition to zonotopes of lower dimension.

    :param zono: zonotope
    :return: list of zonotopes of lower dimension
    """
    # if not isinstance(zono, zonotope):
    #     print(type(zono))
    #     raise TypeError("Input must be a zonotope!")

    dim = zono.dimension

    if dim != 3:
        raise NotImplementedError("This function is only implemented for 3D zonotopes!")

    def normalize_plane_dir(nvec: torch.Tensor) -> torch.Tensor:
        """Normalize plane direction vector.

        :param nvec: plane direction vector
        :return: normalized plane direction vector
        """
        nvec = nvec / torch.norm(nvec)
        if nvec[2] < 0 or (nvec[2] == 0 and nvec[1] < 0):
            nvec = -nvec

        return nvec

    # preparations
    center = zono.center
    generators = zono.generators

    # construct hyperplanes
    hyperplanes: list[set] = []
    norm_vectors: set = set()  # for detecting duplicate plane creation
    for i in range(zono.n_generators):
        # first generator creates a single hyperplane
        if not hyperplanes:
            hyperplanes.append({i})
            if zono.n_generators > 1:
                hyperplanes[0].add(1)
                nvec = torch.cross(generators[0, :], generators[1, :])
                nvec = normalize_plane_dir(nvec)
                norm_vectors.add(tuple(nvec.tolist()))

        for j, hp in enumerate(hyperplanes):
            if i in hp:
                continue

            # check if generator coplanar with existing hyperplane
            if torch.linalg.matrix_rank(generators[[*hp, i], :]) < dim:
                # add generator to hyperplane
                hp.add(i)
                continue

            # create new hyperplane
            for j in hp:  # NOTE: should be a combination in n-d
                # check if hyper plane already exists
                nvec = torch.cross(generators[j, :], generators[i, :])
                nvec = tuple(normalize_plane_dir(nvec).tolist())
                if nvec in norm_vectors:
                    continue

                hyperplanes.append({j, i})
                norm_vectors.add(nvec)

    # early stop: only one hyperplane is found, zonotope itself lies in lower dimension
    if len(hyperplanes) == 1:
        return [zono]

    # calculate norm_vectors
    hp_spans = torch.cat(
        [generators[list(hp)[:2], :].unsqueeze(0) for hp in hyperplanes]
    )
    norm_vectors = torch.cross(hp_spans[:, 0, :], hp_spans[:, 1, :], dim=1)

    # NOTE: summing all other generators up with coefficient 1 or -1,
    # determined by dot product of the incoming generator and the hyperplane normal
    gid_set = set(range(zono.n_generators))
    complement_gens = [gid_set.difference(hp) for hp in hyperplanes]
    centers = torch.ones(
        (2 * len(complement_gens), dim), dtype=zono.dtype
    ) * center.unsqueeze(0)
    for i, cp_ids in enumerate(complement_gens):
        cp_gens = generators[list(cp_ids), :]
        coeffs = ((cp_gens @ norm_vectors[[i], :].T) > 0).float()  # n cp_gens x 1
        coeffs[coeffs == 0] = -1
        center_offset = torch.sum(coeffs * cp_gens, dim=0)
        centers[2 * i, :] += center_offset
        centers[2 * i + 1, :] -= center_offset

    # convert centers and generators to zonotope objects and return
    zonos = []
    for i in range(len(hyperplanes)):
        gens = generators[list(hyperplanes[i]), :]
        zonos.append(zonotope(torch.cat([centers[2 * i, :].unsqueeze(0), gens])))
        zonos.append(zonotope(torch.cat([centers[2 * i + 1, :].unsqueeze(0), gens])))

    return zonos


def plot3d(
    zono: zonotope,
    color: Optional[tuple | np.ndarray] = None,
    alpha: float = 1,
    vertex: bool = True,
    figax=None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a 3D zonotope.

    Args:
        zono (zonotope): zonotope
        color (tuple, optional): color of the zonotope. Defaults to random colors for each face
        alpha (float, optional): transparency of the zonotope. Defaults to 1.
        vertex (bool, optional): whether to plot vertices. Defaults to True.
        figax ([type], optional): figax if reused. Defaults to None.
    """
    # if not isinstance(zono, zonotope):
    #     raise TypeError("Input must be a zonotope!")

    if zono.dimension != 3:
        raise NotImplementedError("This function is only implemented for 3D zonotopes!")

    # faces
    faces = decompose(zono)

    # vertices
    vtx_list = []
    for face in faces:
        vtx_list.append(vertices(face))

    if not figax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = figax

    if color is None:
        color = np.random.rand(5, 3)
    else:
        assert len(color) in (3, 4), "color must be a tuple of length 3 or 4"
        color = np.asarray(color)
    for vtx in vtx_list:
        poly = Poly3DCollection([vtx], alpha=alpha)
        poly.set_facecolor(color)
        ax.add_collection3d(poly)
        if vertex:
            ax.scatter(vtx[:, 0], vtx[:, 1], vtx[:, 2], c="k")

    ax.axis("equal")

    return fig, ax
