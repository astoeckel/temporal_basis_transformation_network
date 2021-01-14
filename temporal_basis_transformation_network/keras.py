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
import tensorflow as tf

from .common import *


class TemporalBasisTrafo(tf.keras.layers.Layer):
    """
    DESCRIPTION
    ===========

    The TemporalBasisTrafo is a TensorFlow Keras layer that implements a fixed
    convolution with a function basis described by a matrix H of size q x N.
    This matrix can be interpreted as a set of q FIR filters of length N. These
    filters can form a discrete temporal basis. Weighting the momentary
    filtered values thus can be used to compute functions over time in a purely
    feed-forward fashion.

    This code is a generalised version of the feed-forward Legendre Memory Unit
    (ff-LMU) model proposed by Narsimha R. Chilkuri in 2020/21. Instead of
    computing the Legendre Delay Network (LND) basis H, this network will work
    witha any temporal basis matrix H. You can for example use the
    "dlop_ldn_function_bases" package [1] to generate basis transformations.

    The network can furthermore be operated in an inverse mode. In this mode,
    if the output of another TemporalBasisTrafo network is provided, the
    network will reconstruct the original input, given that the basis
    transformation matrix H is orthogonal. For example, if H is the DFT matrix
    then a network with inverse = False will compute the forward discrete
    Fourier transformation, whereas a network with inverse = True will compute
    the inverse discrete Fourier transformation.

    Note that this Layer does not have any trainable parameters. It is
    recommended to add a nonlinear layer to the input and the output of this
    network.

    [1] https://github.com/astoeckel/dlop_ldn_function_bases

    INPUT AND OUTPUT DIMENSIONS
    ===========================

    The input and output dimensionalities depend on whether the network is
    operating in forward or inverse mode.

    Forward mode
    ------------

    In forward mode (default), this layer takes an input with dimensions

        [n_1, ..., n_i, M, n_units]           (Input; Mode: Forward)

    The input dimensions n_1, ..., n_i are optional and are interpreted as
    independent batches. M is the number of input samples.

    Depending on the value of "collapse", the output is either an array
    of shape (collapse = True, default)

        [n_1, ..., n_i, M', n_units * q]      (Output; Mode: Forward, Collapse)

    or (collapse = False)

        [n_1, ..., n_i, M', n_units, q]       (Output; Mode: Forward)

    where M' is the number of output samples. If padding is enabled (default)
    then M' = M. Otherwise M' = max(1, M - N + 1).

    Inverse mode
    ------------

    In inverse mode, this layer takes an input with dimensions

        [n_1, ..., n_i, M, n_units * q]       (Input; Mode: Inverse, Pre-Collapse)

    if collapse = True (default), or, if instead collapse = False

        [n_1, ..., n_i, M, n_units, q]        (Input; Mode: Inverse)

    where M is the number of input samples. The output dimensions are

        [n_1, ..., n_i, M, n_units * N]      (Output; Mode: Inverse, Post-Collapse)

    or, if collapse is not set,

        [n_1, ..., n_i, M, n_units,  N]      (Output; Mode: Inverse)

    """
    def __init__(
        self,
        H,
        n_units=1,
        pad=True,
        collapse=True,
        mode=Forward,
        rcond=1e-6,
    ):
        """
        Creates a new instance of the BasisTransformationNetwork using the
        basis transformation in the matrix H.

        Parameters
        ==========

        H:  Basis transformation matrix to use. This must be a 2D array of
            size q x N, where q is the number of dimensions an input
            series is being transformed into, and N is the length of one
            input sequence, or, in other words, the number of timesteps.

        n_units: Number of times the basis transformation unit should be
            repeated.

        pad: In Forward mode, if True, pads the input with zeros such that the
            output size and nput size are exactly the same. That is, if
            pad = True and M samples are fed into the network, M samples
            will be returned.

            Otherwise, if pad = False, if M samples are
            fed into the network then max(1, M - N + 1) samples will be
            returned, where N is the length of the basis transformation
            matrix. Appropriate padding will always be added if
            M - N + 1 < 0.

            This has no effect in Inverse mode, where the number of input and
            output samples will always be the same.

        collapse: Either a single boolean value, or a tuple (pre_collapse,
            post_collapse). A single boolean value b is translated to (b, b).
            The default value is True, applying both a pre- and post-collapse.

            pre_collapse is only relevant when operating in inverse mode, if set
            to true, reverts a post_collapse applied by a previous network.

        mode: Determines the mode the network operates in. Must be set to
            either the "Forward" or "Inverse" constants exported in the
            temporal_basis_transformation_network package.

        rcond: Regularisation constant used when computing the pseudo-inverse of
            H for the inverse mode.
        """
        # Call the inherited constructor
        super().__init__()

        # Make sure the given parameters make sense
        self._q, self._N, self._H, self._n_units, self._pad, self._collapse, \
        self._mode = \
            coerce_params(H, n_units, pad, collapse, mode, rcond, np)

        # Initialize the Tensorflow constants
        self._tf_H, self._tf_pad = None, None

    def get_config(self):
        return {
            "q": self._q,
            "N": self._N,
            "n_units": self._n_units,
            "pad": self._pad,
            "pre_collapse": self._collapse[0],
            "post_collapse": self._collapse[1],
            "mode": repr(self._mode),
        }

    def build(self, S):
        """
        This function is called before the first call to "call". Creates
        TensorFlow constants and computes the input and output
        shapes/permutations.
        """

        # Compute the input/output permutations
        self._input_shape_pre, self._input_perm, self._input_shape_post, \
        self._output_shape_pre, self._output_perm, self._output_shape_post = \
            compute_shapes_and_permutations(
                S, self._n_units, self._q, self._N,
                self._pad, self._collapse, self._mode)

        # Upload the basis transformation H into a tensorflow variable. Reshape
        # the matrix to be compatible with tf.nn.convolution. The first
        # dimension is the number of filters, the second dimension the number of
        # input channels, the third dimension the number of output channels.
        shape = (self._N, 1, self._q) if (self._mode is Forward) else (1, self._q, self._N)
        self._tf_H = tf.constant(value=self._H.T,
                                 shape=shape,
                                 dtype='float32')

        # Padding used to add N - 1 zeros to the beginning of the input array.
        # This way the convolution operation will return exactly N output
        # samples.
        self._tf_pad = None
        if self._mode is Forward:
            M_in, M_out = self._input_shape_post[-2], self._output_shape_pre[-2]
            M_pad = M_out - M_in + self._N - 1
            if M_pad > 0:
                self._tf_pad = tf.constant([[0, 0], [M_pad, 0], [0, 0]],
                                           dtype='int32')

    def call(self, xs):
        """
        Implements the actual basis transformations. Reshapes the inputs,
        computes a convolution, and reshapes the output.
        """
        if not self._input_shape_pre is None:
            xs = tf.reshape(xs, self._input_shape_pre)
        if not self._input_perm is None:
            xs = tf.transpose(xs, perm=self._input_perm)
        if not self._input_shape_post is None:
            xs = tf.reshape(xs, self._input_shape_post)
        if not self._tf_pad is None:
            xs = tf.pad(xs, self._tf_pad)
        if self._mode is Forward:
            ys = tf.nn.convolution(xs, self._tf_H)
        else:
            ys = tf.matmul(xs, self._tf_H)
        if not self._output_shape_pre is None:
            ys = tf.reshape(ys, self._output_shape_pre)
        if not self._output_perm is None:
            ys = tf.transpose(ys, perm=self._output_perm)
        if not self._output_shape_post is None:
            ys = tf.reshape(ys, self._output_shape_post)
        return ys

