#   Temporal Basis Transformation Network
#   Copyright (C) 2020, 2021  Andreas Stöckel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as
#   published by the Free Software Foundation, either version 3 of the
#   License, or (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

from .common import *


def trafo(xs,
          H,
          n_units=1,
          pad=True,
          collapse=True,
          mode=Forward,
          rcond=1e-6,
          np=np):
    """
    Reference implementation of the temporal basis transformation network. This
    implementation does not use TensorFlow, but only numpy. The backing numpy
    implementation can be changed by switching from 
    """
    # Coerce the parameters
    q, N, H, n_units, pad, collapse, mode = coerce_params(
        H, n_units, pad, collapse, mode, rcond, np)

    # Make sure the input data is a float32 array
    xs = np.asarray(xs, dtype=np.float32)

    # Compute the re-shapes that should be applied to the input and output
    input_shape_pre, input_perm, input_shape_post, \
    output_shape_pre, output_perm, output_shape_post = \
        compute_shapes_and_permutations(xs.shape, n_units, q, N, pad, collapse, mode)

    # Re-arrange the input signal
    xs = xs.reshape(input_shape_pre)
    xs = xs.transpose(input_perm)
    xs = xs.reshape(input_shape_post)

    # Pad the input signal
    if mode is Forward:
        M_in, M_out = input_shape_post[-2], output_shape_pre[-2]
        M_pad = M_out - M_in + N - 1
        if M_pad > 0:
            s0, _, s2 = xs.shape
            xs = np.concatenate((np.zeros((s0, M_pad, s2)), xs),
                                axis=1)

    # Compute the convolution
    N_conv = input_shape_post[0]
    if mode is Forward:
        ys = np.zeros((N_conv, M_out, q))
        for i in range(N_conv):
            for j in range(q):
                ys[i, :, j] = np.convolve(xs[i, :, 0], H[j, ::-1], 'valid')
    elif mode is Inverse:
        ys = xs @ H.T

    # Re-arrange the output signal
    ys = ys.reshape(output_shape_pre)
    ys = ys.transpose(output_perm)
    ys = ys.reshape(output_shape_post)

    return ys

