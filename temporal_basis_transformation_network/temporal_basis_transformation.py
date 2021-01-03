#   Temporal Basis Transformation Network
#   Copyright (C) 2020  Andreas Stöckel
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


class TemporalBasisTrafo(tf.keras.layers.Layer):
    """
    The TemporalBasisTrafo is a TensorFlow Keras layer that implements a fixed
    convolution with a function basis described by a matrix H of size q x N.
    This matrix can be interpreted as a set of q FIR filters of length N. These
    filters can form a temporal bases. Weighting the momentary filtered values
    thus can be used to compute functions over time in a purely feed-forward
    fashion.

    This code is a generalised version of the feed-forward Legendre Memory Unit
    (ff-LMU) model proposed by Narsimha R. Chilkuri in 2020. Instead of
    computing the Legendre Delay Network (LND) basis H, this network will work
    witha any temporal basis matrix H. You can for example use the
    "dlop_ldn_function_bases" package [1] to generate basis transformations.

    Note that this Layer does not have any trainable parameters. It is
    recommended to add a nonlinear layer to the input and the output of this
    network.

    This layer takes an input with dimensions

        [n_1, ..., n_i, M, n_units]

    The input dimensions n_1, ..., n_i are optional and are interpreted as
    independent batches. M is the number of input samples.

    The output is an array of shape

        [n_1, ..., n_i, M', n_units, q]

    where M' is the number of output samples. If padding is enabled (default)
    then M' = M. Otherwise M' = max(1, M - N + 1).

    [1] https://github.com/astoeckel/dlop_ldn_function_bases
    """
    def __init__(
        self,
        H,
        n_units=1,
        pad=True,
        strict=False,
    ):
        """
        Creates a new instance of the BasisTransformationNetwork using the
        basis transformation in the matrix H.

        Parameters
        ==========

        H:       Basis transformation matrix to use. This must be a 2D array of
                 size q x N, where q is the number of dimensions an input
                 series is being transformed into, and N is the length of one
                 input sequence, or, in other words, the number of timesteps.

        n_units: Number of times the basis transformation unit should be
                 repeated.

        pad:     If True, pads the input with zeros such that the output size
                 and input size are exactly the same. That is, if pad = True
                 and M samples are fed into the network, M samples will be
                 returned. Otherwise, if pad = False, if M samples are fed into
                 the network then max(1, M - N + 1) samples will be returned,
                 where N is the length of the basis transformation matrix.
                 Appropriate padding will always be added if M - N + 1 < 0.
        """
        # Make sure the given parameters make sense
        self.H = np.asarray(H).astype(dtype=np.float32, copy=True)
        assert self.H.ndim == 2
        assert self.H.shape[0] <= self.H.shape[1]
        assert self.H.shape[0] > 0
        assert self.H.shape[1] > 0
        assert int(n_units) > 0

        # Call the inherited constructor
        super().__init__()

        # Fetch the dimensions q x N from the matrix H and copy the other
        # parameters
        self.q, self.N = self.H.shape
        self.n_units = int(n_units)
        self.pad = bool(pad)

        # Initialize the Tensorflow constants
        self._c_H = None
        self._c_pad = None

    def get_config(self):
        return {
            "q": self.q,
            "N": self.N,
            "n_units": self.n_units
        }

    @staticmethod
    def _intermediate_and_output_shape_and_perms(S, n_units, q, N, pad=True):
        """
        For a given input shape "S" computes the intermediate shape (i.e., the
        shape the input is re-shaped to before passing it into
        tf.nn.convolution), as well as the output shape, i.e., the shape the
        output is reshaped into before returning it. Furthermore returns the
        permutations that need to be applied to the input before reshaping,
        as well as the permutations that need to be applied to the output
        after reshaping.
        """
        assert (len(S) >= 1) and (n_units > 0) and (q > 0) and (N > 0)
        assert (sum(s is None for s in S) <= 1)

        def replace_none(S):
            return (-1 if s is None else s for s in S)

        # Input length
        l = len(S)

        # The last dimension must be equal to the number of units
        if (l < 2) or (S[-1] != n_units):
            fmt = "TemporalBasisTrafo: Invalid shape. " \
                  "Got ({shape}) but expected (..., M_in, {n_units})"
            raise RuntimeError(
                fmt.format(shape=", ".join(map(str, S)), n_units=n_units))

        # Fetch the number of input samples M_in and compute the number of
        # output samples M_out
        M_in, M_out = S[-2], (S[-2] if pad else max(1, S[-2] - N + 1))

        # Compute the intermediate shape. Multiply all input dimensions
        # except for the second-last dimension containing M_in
        n_batch = 1
        for i, s in enumerate(S):
            if i != (l - 2):
                if s is None:
                    n_batch = -1
                    break
                else:
                    n_batch *= s
        intermediate_shape = (n_batch, M_in, 1)

        # Compute the intermediate and output permutation. Unless
        # discard_last is set, the last dimension needs to be placed before
        # the second-last dimension.
        intermediate_perm = tuple(i if i < l - 2 else (2 * l - 3 - i)
                                  for i in range(l))

        # Compute the output shape and the output permutation
        output_shape = tuple((*replace_none(S[:-2]), n_units, M_out, q))
        output_perm = tuple((*range(0, l - 2), l - 1, l - 2, l))

        return intermediate_shape, intermediate_perm, \
               output_shape, output_perm

    def build(self, input_shape):
        """
        This function is called before the first call to "call". Computes the
        intermediate and output shape to be used in the computations below.
        """
        self._intermediate_shape, self._intermediate_perm, \
        self._output_shape, self._output_perm = \
            TemporalBasisTrafo._intermediate_and_output_shape_and_perms(
                input_shape, self.n_units, self.q, self.N, self.pad)

        # Upload the basis transformation H into a tensorflow variable. Reshape
        # the matrix to be compatible with tf.nn.convolution. The first
        # dimension is the number of filters, the second dimension the number of
        # input channels, the third dimension the number of output channels.
        self._c_H = tf.constant(
            value=self.H.T,
            shape=(self.N, 1, self.q),
            dtype='float32',
        )

        # Padding used to add N - 1 zeros to the beginning of the input array.
        # This way the convolution operation will return exactly N output
        # samples.
        M_in, M_out = self._intermediate_shape[-2], self._output_shape[-2]
        M_pad = M_out - M_in + self.N - 1
        if M_pad == 0:
            self._c_pad = None
        else:
            self._c_pad = tf.constant([[0, 0], [M_pad, 0], [0, 0]],
                                      dtype='int32')


    def call(self, xs):
        # Reshape the input into the shape required by tf.nn.convolution.
        # This will give us an array of shape (n_batch * n_units, N, 1), where
        # "n_batch" is a product of all "extra" dimensions.
        xs_transposed = tf.transpose(xs, perm=self._intermediate_perm)
        xs_reshaped = tf.reshape(xs_transposed, self._intermediate_shape)

        # Pad the second input dimension such that the desired number of output
        # dimensions is reached. If _c_pad is None, then no padding is
        # required.
        if self._c_pad is None:
            xs_padded = xs_reshaped
        else:
            xs_padded = tf.pad(xs_reshaped, self._c_pad, name='basis_trafo_pad')

        # Compute the convolution with the basis transformation matrix H.
        ys = tf.nn.convolution(xs_padded, self._c_H, name='basis_trafo_conv')

        # Now reshape the output to the output dimensions we computed above and
        # permute the dimensions back so they match the desired output shape.
        ys_reshaped = tf.reshape(ys, self._output_shape)
        ys_transposed = tf.transpose(ys_reshaped, perm=self._output_perm)

        # Done!
        return ys_transposed

