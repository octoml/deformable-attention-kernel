import torch
import torch.nn.functional as F
import tvm
from tvm.script import tir as T
import numpy as np
import timeit
import MultiScaleDeformableAttention as MSDA
import argparse


@torch.no_grad()
def deformable_attention_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """Pytorch implementation of deformable attention from
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/functions/ms_deform_attn_func.py"""
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = (
            value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        )
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        N_ * M_, 1, Lq_, L_ * P_
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(N_, M_ * D_, Lq_)
    )
    return output.transpose(1, 2).contiguous()


"""
Deformable Attention

N = batch size
L = number of grids to sample from ("levels")
Lq = "query length"
S = sum of width * height of samples in value_spatial_shapes
I = number of samples
M = number of attention heads
P = number of points to sample
D = depth

This function computes
`output[N, Lq, M * D] = sum_P value(N, sampling_locations[N, Lq, M, L, P], M, D) * attention_weights[N, Lq, M, L, P]`
where `value()` denotes a bilinear interpolation of sampling from value as if
it were a grid. This operation is memory bound.

For the CPU implementation, we parallelize over the Lq dimension (though we
could also parallelize over the batch dimension too). For each corner in
bilinear interpolation, we compute `value(sampling_locations[])` in chunks over
M and D to maximize locality (M, D are the innermost dimensions of `value`).

For the GPU implementation we assign threads to load 4 element chunks in the D
dimension (optimal load size is 128 = 4 * 32). We assign blocks to the Lq, M,
and N dimensions so that we have enough concurrent work to saturate the SMs.
The GPU implementation achieves 80% of maximum bandwidth, which is pretty good.

Parameters
----------
value: array[float, N, S, M, D]
grids we are sampling from

value_spacial_shapes: array[int, L, 2]
size of the grids to sample from

sampling_locations: array[float, N, Lq, M, L, P, 2]
sampling_locations is in [0,1]

attention_weights: array[float, N, Lq, M, L, P]

Returns
-------
output: array[float, N, Lq, M*D]
"""


# Reference implementation, not optimized
@T.prim_func
def deformable_attention_tvmscript_reference(
    value: T.handle,
    value_spatial_shapes: T.handle,
    sampling_locations: T.handle,
    attention_weights: T.handle,
    output: T.handle,
) -> None:
    n = T.var("int32")
    l = T.var("int32")
    lq = T.var("int32")
    s = T.var("int32")
    m = T.var("int32")
    d = T.var("int32")
    p = T.var("int32")
    value_ = T.match_buffer(value, [n, s, m, d], dtype="float32")
    value_spatial_shapes_ = T.match_buffer(value_spatial_shapes, [l, 2], dtype="int32")
    sampling_locations_ = T.match_buffer(
        sampling_locations, [n, lq, m, l, p, 2], dtype="float32"
    )
    attention_weights_ = T.match_buffer(
        attention_weights, [n, lq, m, l, p], dtype="float32"
    )
    output_ = T.match_buffer(output, [n, lq, m * d], dtype="float32")

    # These are temporaries used to store information
    value_offset = T.alloc_buffer([1], dtype="int32", scope="local")
    attention_sum = T.alloc_buffer([1], dtype="float32", scope="local")
    height_width = T.alloc_buffer([2], dtype="int32", scope="local")
    xy = T.alloc_buffer([2], dtype="float32", scope="local")
    xy_grid = T.alloc_buffer([2], dtype="float32", scope="local")
    xy_rounded = T.alloc_buffer(
        [2, 2], dtype="int32", scope="local"
    )  # First dim is x,y second is floor, ceil
    corner_values = T.alloc_buffer([2, 2], dtype="float32", scope="local")

    for batch in range(n):
        for i_m in range(m):
            for i_d in range(d):
                for j in range(lq):
                    attention_sum[0] = 0.0
                    for i in range(l):
                        value_offset[0] = 0
                        for ii in range(i):
                            value_offset[0] += (
                                value_spatial_shapes_[ii, 0]
                                * value_spatial_shapes_[ii, 1]
                            )
                        for k in range(p):
                            # The sampling grid is in the range 0, 1. We convert it to
                            # [-0.5, (height|width) - 0.5]. This offset is
                            # supposed to make interpolation resolution
                            # independent.
                            height_width[0] = value_spatial_shapes_[i, 0]
                            height_width[1] = value_spatial_shapes_[i, 1]
                            xy[1] = sampling_locations_[batch, j, i_m, i, k, 0]
                            xy[0] = sampling_locations_[batch, j, i_m, i, k, 1]
                            # Convert x,y to indices in the grid
                            xy_grid[0] = (
                                xy[0] * T.cast(height_width[0], "float32") - 0.5
                            )
                            xy_grid[1] = (
                                xy[1] * T.cast(height_width[1], "float32") - 0.5
                            )
                            # Get 4 integer locations surrounding x_grid, y_grid. Dims: x,y then floor, ceil
                            xy_rounded[0, 0] = T.cast(
                                T.floor(xy_grid[0], dtype="float32"), "int32"
                            )
                            xy_rounded[0, 1] = xy_rounded[0, 0] + 1
                            xy_rounded[1, 0] = T.cast(
                                T.floor(xy_grid[1], dtype="float32"), "int32"
                            )
                            xy_rounded[1, 1] = xy_rounded[1, 0] + 1

                            # This next series of statements performs the
                            # lookups of the four grid aligned points
                            # surrounding the point we will interpolate
                            if (
                                xy_rounded[0, 0] < 0
                                or xy_rounded[0, 0] >= height_width[0]
                                or xy_rounded[1, 0] < 0
                                or xy_rounded[1, 0] >= height_width[1]
                            ):
                                corner_values[0, 0] = 0.0
                            else:
                                corner_values[0, 0] = value_[
                                    batch,
                                    value_offset[0]
                                    + xy_rounded[0, 0] * height_width[1]
                                    + xy_rounded[1, 0],
                                    i_m,
                                    i_d,
                                ]
                            if (
                                xy_rounded[0, 1] < 0
                                or xy_rounded[0, 1] >= height_width[0]
                                or xy_rounded[1, 0] < 0
                                or xy_rounded[1, 0] >= height_width[1]
                            ):
                                corner_values[1, 0] = 0.0
                            else:
                                corner_values[1, 0] = value_[
                                    batch,
                                    value_offset[0]
                                    + xy_rounded[0, 1] * height_width[1]
                                    + xy_rounded[1, 0],
                                    i_m,
                                    i_d,
                                ]
                            if (
                                xy_rounded[0, 0] < 0
                                or xy_rounded[0, 0] >= height_width[0]
                                or xy_rounded[1, 1] < 0
                                or xy_rounded[1, 1] >= height_width[1]
                            ):
                                corner_values[0, 1] = 0.0
                            else:
                                corner_values[0, 1] = value_[
                                    batch,
                                    value_offset[0]
                                    + xy_rounded[0, 0] * height_width[1]
                                    + xy_rounded[1, 1],
                                    i_m,
                                    i_d,
                                ]
                            if (
                                xy_rounded[0, 1] < 0
                                or xy_rounded[0, 1] >= height_width[0]
                                or xy_rounded[1, 1] < 0
                                or xy_rounded[1, 1] >= height_width[1]
                            ):
                                corner_values[1, 1] = 0.0
                            else:
                                corner_values[1, 1] = value_[
                                    batch,
                                    value_offset[0]
                                    + xy_rounded[0, 1] * height_width[1]
                                    + xy_rounded[1, 1],
                                    i_m,
                                    i_d,
                                ]
                            # bilinear interpolation
                            attention_sum[0] += (
                                corner_values[0, 0]
                                * (T.cast(xy_rounded[0, 1], "float32") - xy_grid[0])
                                * (T.cast(xy_rounded[1, 1], "float32") - xy_grid[1])
                                + corner_values[1, 0]
                                * (xy_grid[0] - T.cast(xy_rounded[0, 0], "float32"))
                                * (T.cast(xy_rounded[1, 1], "float32") - xy_grid[1])
                                + corner_values[0, 1]
                                * (T.cast(xy_rounded[0, 1], "float32") - xy_grid[0])
                                * (xy_grid[1] - T.cast(xy_rounded[1, 0], "float32"))
                                + corner_values[1, 1]
                                * (xy_grid[0] - T.cast(xy_rounded[0, 0], "float32"))
                                * (xy_grid[1] - T.cast(xy_rounded[1, 0], "float32"))
                            ) * attention_weights_[batch, j, i_m, i, k]
                    output_[batch, j, i_m * d + i_d] = attention_sum[0]


# optimized implementation for CPU
@T.prim_func
def deformable_attention_tvmscript(
    value: T.handle,
    value_spatial_shapes: T.handle,
    sampling_locations: T.handle,
    attention_weights: T.handle,
    output: T.handle,
) -> None:
    n = T.var("int32")  # number of batches
    l = T.var("int32")  # number of levels
    lq = T.var("int32")  # query length
    s = T.var("int32")  # combined height*width of every grid
    m = T.var("int32")  # number of attention heads
    d = T.var("int32")  # depth
    p = T.var("int32")  # number of points to sample

    # We need to bind handles to buffers so that we can access elements from them.
    value_ = T.match_buffer(value, [n, s, m, d], dtype="float32")
    value_spatial_shapes_ = T.match_buffer(value_spatial_shapes, [l, 2], dtype="int32")
    sampling_locations_ = T.match_buffer(
        sampling_locations, [n, lq, m, l, p, 2], dtype="float32"
    )
    attention_weights_ = T.match_buffer(
        attention_weights, [n, lq, m, l, p], dtype="float32"
    )
    output_ = T.match_buffer(output, [n, lq, m * d], dtype="float32")

    # These are temporaries used during the computation. They are stored in
    # thread local registers.
    value_offset = T.alloc_buffer([1], dtype="int32", scope="local")
    attention_sum = T.alloc_buffer([m, d], dtype="float32", scope="local")
    height_width = T.alloc_buffer([2], dtype="int32", scope="local")
    xy = T.alloc_buffer([2], dtype="float32", scope="local")
    xy_grid = T.alloc_buffer([2, m], dtype="float32", scope="local")
    xy_rounded = T.alloc_buffer(
        [2, 2, m], dtype="int32", scope="local"
    )  # first dim is x,y second is floor, ceil
    corner_values = T.alloc_buffer([2, 2, m, d], dtype="float32", scope="local")

    for j in T.parallel(0, lq):
        for batch in range(n):
            for i_d in range(d):
                for i_m in range(m):
                    attention_sum[i_m, i_d] = 0.0
            for i in range(l):
                value_offset[0] = 0
                for ii in range(i):
                    value_offset[0] += (
                        value_spatial_shapes_[ii, 0] * value_spatial_shapes_[ii, 1]
                    )
                for k in range(p):
                    for i_m in range(m):
                        # the sampling grid is in the range 0, 1. We convert it to
                        # [-0.5, (height|width) - 0.5].
                        height_width[0] = value_spatial_shapes_[i, 0]
                        height_width[1] = value_spatial_shapes_[i, 1]
                        xy[1] = sampling_locations_[batch, j, i_m, i, k, 0]
                        xy[0] = sampling_locations_[batch, j, i_m, i, k, 1]
                        # Convert x,y to indices in the grid.
                        xy_grid[0, i_m] = (
                            xy[0] * T.cast(height_width[0], "float32") - 0.5
                        )
                        xy_grid[1, i_m] = (
                            xy[1] * T.cast(height_width[1], "float32") - 0.5
                        )
                        # Get 4 integer locations surrounding x_grid, y_grid. Dims: x,y then floor, ceil
                        xy_rounded[0, 0, i_m] = T.cast(
                            T.floor(xy_grid[0, i_m], dtype="float32"), "int32"
                        )
                        xy_rounded[0, 1, i_m] = xy_rounded[0, 0, i_m] + 1
                        xy_rounded[1, 0, i_m] = T.cast(
                            T.floor(xy_grid[1, i_m], dtype="float32"), "int32"
                        )
                        xy_rounded[1, 1, i_m] = xy_rounded[1, 0, i_m] + 1

                    # Look up 4 grid locations surrounding the sampling
                    # point. Use 0 if the grid locations are out of
                    # bounds.
                    for i_m in range(m):
                        if (
                            xy_rounded[0, 0, i_m] < 0
                            or xy_rounded[0, 0, i_m] >= height_width[0]
                            or xy_rounded[1, 0, i_m] < 0
                            or xy_rounded[1, 0, i_m] >= height_width[1]
                        ):
                            for i_d in range(d):
                                corner_values[0, 0, i_m, i_d] = 0.0
                        else:
                            for i_d in range(d):
                                corner_values[0, 0, i_m, i_d] = value_[
                                    batch,
                                    value_offset[0]
                                    + xy_rounded[0, 0, i_m] * height_width[1]
                                    + xy_rounded[1, 0, i_m],
                                    i_m,
                                    i_d,
                                ]
                    for i_m in range(m):
                        if (
                            xy_rounded[0, 0, i_m] < 0
                            or xy_rounded[0, 0, i_m] >= height_width[0]
                            or xy_rounded[1, 1, i_m] < 0
                            or xy_rounded[1, 1, i_m] >= height_width[1]
                        ):
                            for i_d in range(d):
                                corner_values[0, 1, i_m, i_d] = 0.0
                        else:
                            for i_d in range(d):
                                corner_values[0, 1, i_m, i_d] = value_[
                                    batch,
                                    value_offset[0]
                                    + xy_rounded[0, 0, i_m] * height_width[1]
                                    + xy_rounded[1, 1, i_m],
                                    i_m,
                                    i_d,
                                ]
                    for i_m in range(m):
                        if (
                            xy_rounded[0, 1, i_m] < 0
                            or xy_rounded[0, 1, i_m] >= height_width[0]
                            or xy_rounded[1, 0, i_m] < 0
                            or xy_rounded[1, 0, i_m] >= height_width[1]
                        ):
                            for i_d in range(d):
                                corner_values[1, 0, i_m, i_d] = 0.0
                        else:
                            for i_d in range(d):
                                corner_values[1, 0, i_m, i_d] = value_[
                                    batch,
                                    value_offset[0]
                                    + xy_rounded[0, 1, i_m] * height_width[1]
                                    + xy_rounded[1, 0, i_m],
                                    i_m,
                                    i_d,
                                ]
                    for i_m in range(m):
                        if (
                            xy_rounded[0, 1, i_m] < 0
                            or xy_rounded[0, 1, i_m] >= height_width[0]
                            or xy_rounded[1, 1, i_m] < 0
                            or xy_rounded[1, 1, i_m] >= height_width[1]
                        ):
                            for i_d in range(d):
                                corner_values[1, 1, i_m, i_d] = 0.0
                        else:
                            for i_d in range(d):
                                corner_values[1, 1, i_m, i_d] = value_[
                                    batch,
                                    value_offset[0]
                                    + xy_rounded[0, 1, i_m] * height_width[1]
                                    + xy_rounded[1, 1, i_m],
                                    i_m,
                                    i_d,
                                ]

                    for i_m in range(m):
                        for i_d in range(d):
                            # bilinear interpolation
                            attention_sum[i_m, i_d] += (
                                corner_values[0, 0, i_m, i_d]
                                * (
                                    T.cast(xy_rounded[0, 1, i_m], "float32")
                                    - xy_grid[0, i_m]
                                )
                                * (
                                    T.cast(xy_rounded[1, 1, i_m], "float32")
                                    - xy_grid[1, i_m]
                                )
                                + corner_values[1, 0, i_m, i_d]
                                * (
                                    xy_grid[0, i_m]
                                    - T.cast(xy_rounded[0, 0, i_m], "float32")
                                )
                                * (
                                    T.cast(xy_rounded[1, 1, i_m], "float32")
                                    - xy_grid[1, i_m]
                                )
                                + corner_values[0, 1, i_m, i_d]
                                * (
                                    T.cast(xy_rounded[0, 1, i_m], "float32")
                                    - xy_grid[0, i_m]
                                )
                                * (
                                    xy_grid[1, i_m]
                                    - T.cast(xy_rounded[1, 0, i_m], "float32")
                                )
                                + corner_values[1, 1, i_m, i_d]
                                * (
                                    xy_grid[0, i_m]
                                    - T.cast(xy_rounded[0, 0, i_m], "float32")
                                )
                                * (
                                    xy_grid[1, i_m]
                                    - T.cast(xy_rounded[1, 0, i_m], "float32")
                                )
                            ) * attention_weights_[batch, j, i_m, i, k]
                    for i_m in range(m):
                        for i_d in range(d):
                            output_[batch, j, i_m * d + i_d] = attention_sum[i_m, i_d]


# optimized implementation for GPU
@T.prim_func
def deformable_attention_tvmscript_gpu(
    value: T.handle,
    value_spatial_shapes: T.handle,
    value_level_start_index: T.handle,
    sampling_locations: T.handle,
    attention_weights: T.handle,
    output: T.handle,
) -> None:
    n = T.var("int32")
    l = T.var("int32")
    lq = T.var("int32")
    s = T.var("int32")
    m = T.var("int32")
    d = T.var("int32")
    p = T.var("int32")
    value_ = T.match_buffer(value, [n, s, m, d], dtype="float32")
    value_spatial_shapes_ = T.match_buffer(value_spatial_shapes, [l, 2], dtype="int32")
    value_level_start_index_ = T.match_buffer(
        value_level_start_index, [l], dtype="int32"
    )
    sampling_locations_ = T.match_buffer(
        sampling_locations, [n, lq, m, l, p, 2], dtype="float32"
    )
    attention_weights_ = T.match_buffer(
        attention_weights, [n, lq, m, l, p], dtype="float32"
    )
    output_ = T.match_buffer(output, [n, lq, m * d], dtype="float32")

    # Each thread iterates over 4 elements in dimension D at a time for optimal load size.
    assert d >= 4, "D must be at least 4"

    # These are temporaries used to store information
    attention_sum = T.alloc_buffer([2], dtype="float32", scope="local")
    height_width = T.alloc_buffer([2], dtype="int32", scope="local")
    xy = T.alloc_buffer([2], dtype="float32", scope="local")
    xy_grid = T.alloc_buffer([2], dtype="float32", scope="local")
    xy_rounded = T.alloc_buffer(
        [2, 2], dtype="int32", scope="local"
    )  # first dim is x,y second is floor, ceil
    corner_values = T.alloc_buffer([2, 2, 4], dtype="float32", scope="local")

    for io_d in T.thread_binding(0, d // 4, thread="threadIdx.x"):
        for j in T.thread_binding(0, lq, thread="blockIdx.x"):
            for batch in T.thread_binding(0, n, thread="blockIdx.y"):
                for i_m in T.thread_binding(0, m, thread="blockIdx.z"):
                    for ii_d in range(4):
                        attention_sum[ii_d] = 0.0
                    for i in range(l):
                        height_width[0] = value_spatial_shapes_[i, 0]
                        height_width[1] = value_spatial_shapes_[i, 1]
                        for k in range(p):
                            # the sampling grid is in the range 0, 1. We convert it to
                            # [-0.5, (height|width) - 0.5].
                            xy[1] = sampling_locations_[batch, j, i_m, i, k, 0]
                            xy[0] = sampling_locations_[batch, j, i_m, i, k, 1]
                            # convert x,y to indices in the grid
                            xy_grid[0] = (
                                xy[0] * T.cast(height_width[0], "float32") - 0.5
                            )
                            xy_grid[1] = (
                                xy[1] * T.cast(height_width[1], "float32") - 0.5
                            )
                            # get 4 integer locations surrounding x_grid, y_grid. Dims: x,y then floor, ceil
                            xy_rounded[0, 0] = T.cast(
                                T.floor(xy_grid[0], dtype="float32"), "int32"
                            )
                            xy_rounded[0, 1] = xy_rounded[0, 0] + 1
                            xy_rounded[1, 0] = T.cast(
                                T.floor(xy_grid[1], dtype="float32"), "int32"
                            )
                            xy_rounded[1, 1] = xy_rounded[1, 0] + 1

                            # Look up 4 grid locations surrounding the sampling
                            # point. Use 0 if the grid locations are out of
                            # bounds.
                            if (
                                xy_rounded[0, 0] < 0
                                or xy_rounded[0, 0] >= height_width[0]
                                or xy_rounded[1, 0] < 0
                                or xy_rounded[1, 0] >= height_width[1]
                            ):
                                for ii_d in range(4):
                                    corner_values[0, 0, ii_d] = 0.0
                            else:
                                for ii_d in range(4):
                                    corner_values[0, 0, ii_d] = value_[
                                        batch,
                                        value_level_start_index_[i]
                                        + xy_rounded[0, 0] * height_width[1]
                                        + xy_rounded[1, 0],
                                        i_m,
                                        io_d * 4 + ii_d,
                                    ]
                            if (
                                xy_rounded[0, 0] < 0
                                or xy_rounded[0, 0] >= height_width[0]
                                or xy_rounded[1, 1] < 0
                                or xy_rounded[1, 1] >= height_width[1]
                            ):
                                for ii_d in range(4):
                                    corner_values[0, 1, ii_d] = 0.0
                            else:
                                for ii_d in range(4):
                                    corner_values[0, 1, ii_d] = value_[
                                        batch,
                                        value_level_start_index_[i]
                                        + xy_rounded[0, 0] * height_width[1]
                                        + xy_rounded[1, 1],
                                        i_m,
                                        io_d * 4 + ii_d,
                                    ]
                            if (
                                xy_rounded[0, 1] < 0
                                or xy_rounded[0, 1] >= height_width[0]
                                or xy_rounded[1, 0] < 0
                                or xy_rounded[1, 0] >= height_width[1]
                            ):
                                for ii_d in range(4):
                                    corner_values[1, 0, ii_d] = 0.0
                            else:
                                for ii_d in range(4):
                                    corner_values[1, 0, ii_d] = value_[
                                        batch,
                                        value_level_start_index_[i]
                                        + xy_rounded[0, 1] * height_width[1]
                                        + xy_rounded[1, 0],
                                        i_m,
                                        io_d * 4 + ii_d,
                                    ]
                            if (
                                xy_rounded[0, 1] < 0
                                or xy_rounded[0, 1] >= height_width[0]
                                or xy_rounded[1, 1] < 0
                                or xy_rounded[1, 1] >= height_width[1]
                            ):
                                for ii_d in range(4):
                                    corner_values[1, 1, ii_d] = 0.0
                            else:
                                for ii_d in range(4):
                                    corner_values[1, 1, ii_d] = value_[
                                        batch,
                                        value_level_start_index_[i]
                                        + xy_rounded[0, 1] * height_width[1]
                                        + xy_rounded[1, 1],
                                        i_m,
                                        io_d * 4 + ii_d,
                                    ]

                            for ii_d in range(4):
                                # bilinear interpolation
                                attention_sum[ii_d] += (
                                    corner_values[0, 0, ii_d]
                                    * (T.cast(xy_rounded[0, 1], "float32") - xy_grid[0])
                                    * (T.cast(xy_rounded[1, 1], "float32") - xy_grid[1])
                                    + corner_values[1, 0, ii_d]
                                    * (xy_grid[0] - T.cast(xy_rounded[0, 0], "float32"))
                                    * (T.cast(xy_rounded[1, 1], "float32") - xy_grid[1])
                                    + corner_values[0, 1, ii_d]
                                    * (T.cast(xy_rounded[0, 1], "float32") - xy_grid[0])
                                    * (xy_grid[1] - T.cast(xy_rounded[1, 0], "float32"))
                                    + corner_values[1, 1, ii_d]
                                    * (xy_grid[0] - T.cast(xy_rounded[0, 0], "float32"))
                                    * (xy_grid[1] - T.cast(xy_rounded[1, 0], "float32"))
                                ) * attention_weights_[batch, j, i_m, i, k]
                    for ii_d in range(4):
                        output_[batch, j, i_m * d + io_d * 4 + ii_d] = attention_sum[
                            ii_d
                        ]


def benchmark(target, value, shapes, sampling_locations, attention_weights):
    # Kernel is specialized to that shapes become concrete values
    specialized = deformable_attention_tvmscript.specialize(
        dict(
            zip(
                deformable_attention_tvmscript.params,
                [
                    tvm.tir.decl_buffer(x.shape)
                    for x in [value, shapes, sampling_locations, attention_weights]
                ],
            )
        )
    )
    f = tvm.build(specialized, target=target, name="deformable_attention_tvmscript")
    output = tvm.nd.array(
        np.zeros(
            (
                value.shape[0],
                sampling_locations.shape[1],
                value.shape[2] * value.shape[3],
            ),
            "float32",
        )
    )
    timer = f.time_evaluator(f.entry_name, tvm.cpu(), number=10, repeat=10)
    report = timer(
        tvm.nd.array(value),
        tvm.nd.array(shapes.int()),
        tvm.nd.array(sampling_locations),
        tvm.nd.array(attention_weights),
        output,
    )
    print(f"TVMScript CPU: {report.mean*1000:.2f}ms")

    time_ms = (
        timeit.timeit(
            lambda: deformable_attention_pytorch(
                value, shapes, sampling_locations, attention_weights
            ),
            number=100,
        )
        / 100
        * 1000
    )
    print(f"PyTorch: {time_ms:.2f}ms")
    print(f"Speedup: {time_ms/(report.mean*1000):.2f}x")

    # Check for correctness
    torch_da = deformable_attention_pytorch(
        value, shapes, sampling_locations, attention_weights
    )
    np.testing.assert_allclose(output.numpy(), torch_da.numpy(), atol=1e-6, rtol=1e-6)


def benchmark_gpu(
    target,
    value,
    shapes,
    value_level_start_index,
    sampling_locations,
    attention_weights,
):
    # Kernel is specialized to that shapes become concrete values
    specialized = deformable_attention_tvmscript_gpu.specialize(
        dict(
            zip(
                deformable_attention_tvmscript_gpu.params,
                [
                    tvm.tir.decl_buffer(x.shape)
                    for x in [
                        value,
                        shapes,
                        value_level_start_index,
                        sampling_locations,
                        attention_weights,
                    ]
                ],
            )
        )
    )
    f = tvm.build(specialized, target=target, name="deformable_attention_tvmscript_gpu")
    dev = tvm.cuda()
    output = tvm.nd.array(
        np.zeros(
            (
                value.shape[0],
                sampling_locations.shape[1],
                value.shape[2] * value.shape[3],
            ),
            "float32",
        ),
        device=dev,
    )
    timer = f.time_evaluator(f.entry_name, dev, number=10, repeat=10)
    report = timer(
        tvm.nd.array(value, device=dev),
        tvm.nd.array(shapes.int(), device=dev),
        tvm.nd.array(level_start_index.int(), device=dev),
        tvm.nd.array(sampling_locations, device=dev),
        tvm.nd.array(attention_weights, device=dev),
        output,
    )
    print(f"TVMScript GPU: {report.mean*1000:.3f}ms")

    value_pt = value.cuda()
    shapes_pt = shapes.cuda()
    value_level_start_index_pt = value_level_start_index.cuda()
    sampling_locations_pt = sampling_locations.cuda()
    attention_weights_pt = attention_weights.cuda()

    def f():
        MSDA.ms_deform_attn_forward(
            value_pt,
            shapes_pt,
            value_level_start_index_pt,
            sampling_locations_pt,
            attention_weights_pt,
            64,
        )
        # necessary because kernel launches are async
        torch.cuda.synchronize()

    time_ms = timeit.timeit(f, number=100) / 100 * 1000
    print(f"Handwritten CUDA: {time_ms:.3f}ms")
    print(f"Speedup: {time_ms/(report.mean*1000):.2f}x")

    # Check for correctness
    torch_da = deformable_attention_pytorch(
        value, shapes, sampling_locations, attention_weights
    )
    np.testing.assert_allclose(output.numpy(), torch_da.numpy(), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "target",
        help="Target to run against. Use `llvm -mcpu=skylake` for CPU (assuming Intel Skylake CPU) and `cuda` for GPU.",
    )
    args = arg_parser.parse_args()
    target = tvm.target.Target(args.target)

    for batch_size in [1, 4, 8]:
        print(f"BATCH SIZE {batch_size}")
        # These values are taken from https://github.com/fundamentalvision/Deformable-DETR
        N, M, D = (
            batch_size,
            8,
            256,
        )  # batch size, number of heads, depth
        Lq, L, P = (
            100,
            4,
            4,
        )  # query length, levels, points to sample. models/deformable_detr.py says 100 length query for COCO.
        shapes = torch.as_tensor(
            [[84, 117], [42, 59], [21, 30], [11, 15]], dtype=torch.long
        )
        level_start_index = torch.cat(
            (shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1])
        )
        S = sum([(H * W).item() for H, W in shapes])

        value = torch.rand(N, S, M, D) * 0.01
        sampling_locations = torch.rand(N, Lq, M, L, P, 2)
        attention_weights = torch.rand(N, Lq, M, L, P) + 1e-5
        attention_weights /= attention_weights.sum(-1, keepdim=True).sum(
            -2, keepdim=True
        )

        if str(target.kind) == "llvm":
            benchmark(target, value, shapes, sampling_locations, attention_weights)
            print()
        elif str(target.kind) == "cuda":
            benchmark_gpu(
                target,
                value,
                shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
            )
            print()
        else:
            print(f"Unknown target {args.target}")
