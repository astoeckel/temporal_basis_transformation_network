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

import tensorflow as tf
import tensorflow.keras as keras


class TemporalBasisTrafo(keras.layers.Layer):
    """
    The TemporalBasisTrafo is a Keras layer that implements a fixed convolution
    with a function basis described by a matrix H of size q x N. This matrix
    can be interpreted as a set of FIR filters that form a temporal basis.

    This is a generalised version of the feed-forward Legendre Delay Network
    model proposed by Narsimha R. Chilkuri in 2020. Instead of computing the
    Legendre Delay Network basis H, this network will work with any basis
    matrix H. You can for example use the "dlop_ldn_function_bases" package [1]
    to generate basis transformations.

    Note that this Layer does not have any trainable parameters. It is
    recommended to add a nonlinear layer to the input and the output of this
    network.

    This layer takes an input with dimensions

        [n_1, ..., n_i, N, n_units]

    The input dimensions n_1, ..., n_i are optional and are interpreted as
    independent batches. n_units can be skipped if n_units = 1. N corresponds
    to the number of samples the convolution is defined over.

    The output is an array of shape

        [n_1, ..., n_i, N, n_units, q]

    If "n_units" was skipped in the input, it will also skipped in the output.

    [1] https://github.com/astoeckel/dlop_ldn_function_bases
    """
    def __init__(
        self,
        H,
        n_units=1,
    ):
        """
        Creates a new instance of the BasisTransformationNetwork using the
        basis transformation in the matrix H.

        Parameters
        ==========

        H:       Basis transformation to use. This must be a 2D array matrix of
                 size q x N, where q is the number of dimensions an input
                 series is being transformed into, and N is the length of one
                 input sequence, or, in other words, the number of timesteps.
        n_units: Number of times the basis transformation unit should be
                 repeated.
        """
        # Make sure the given parameters make sense
        assert H.ndim == 2
        assert n_units > 0

        # Call the inherited constructor
        super().__init__()

        # Fetch the dimensions q x N from the matrix H and copy the other
        # parameters
        self.q, self.N = H.shape
        self.n_units = n_units

        # Upload the basis transformation H into a tensorflow variable. Reshape
        # the matrix to be compatible with tf.nn.convolution. The first
        # dimension is the number of filters, the second dimension the number of
        # input channels, the third dimension the number of output channels.
        self.H = tf.constant(
            value=H.T,
            shape=(self.N, 1, self.q),
            dtype='float32',
        )

        # Padding used to add N - 1 zeros to the beginning of the input array.
        # This way the convolution operation will return exactly N output
        # samples.
        self.pad = tf.constant([[0, 0], [self.N - 1, 0], [0, 0]],
                               dtype='int32')

    @staticmethod
    def _intermediate_and_output_shape_and_perms(S, n_units, q, N):
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

        def _impl(S, discard_units):
            # Input length
            l = len(S)

            # Make sure the two last dimensions are correct
            if (S[-1] != n_units) or (S[-2] != N) or (l < 2):
                return None

            # Compute the intermediate shape. Multiply all input dimensions
            # except for the second-last dimension containing N.
            n_batch = 1
            for i, s in enumerate(S):
                if i != (l - 2):
                    n_batch *= s
            intermediate_shape = (n_batch, N, 1)

            # Compute the intermediate and output permutation. Unless
            # discard_last is set, the last dimension needs to be placed before
            # the second-last dimension.
            if discard_units:
                intermediate_perm = tuple(range(l - 1))
            else:
                intermediate_perm = tuple(i if i < l - 2 else (2 * l - 3 - i)
                                          for i in range(l))

            # Compute the output shape and the output permutation
            if discard_units:
                output_shape = tuple((*S[:-2], N, q))
                output_perm = tuple(range(l))
            else:
                output_shape = tuple((*S[:-2], n_units, N, q))
                output_perm = tuple((*range(0, l - 2), l - 1, l - 2, l))

            return intermediate_shape, intermediate_perm, \
                   output_shape, output_perm

        # If we have more than two input dimensions, try to directly interpret
        # the input dimensions
        res = None
        if (res is None) and (len(S) >= 2):
            res = _impl(S, False)

        # If that didn't work, and the number of units is one, artificially add
        # a one to the given shape
        if (res is None) and (len(S) >= 1) and (n_units == 1):
            res = _impl((*S, 1), True)

        # If neither of these two options worked, something went wrong. Raise an
        # exception!
        if (res is None):
            fmt = "TemporalBasisTrafo: Invalid shape. " \
                  "Got ({shape}) but expected (..., {N}, {n_units})"
            if n_units == 1:
                fmt += " or (..., {N})"
            raise RuntimeError(
                fmt.format(shape=", ".join(map(str, S)), N=N, n_units=n_units))

        return res

    def build(self, input_shape):
        """
        This function is called before the first call to "call". Computes the
        intermediate and output shape to be used in the computations below.
        """
        self._intermediate_shape, self._intermediate_perm, \
        self._output_shape, self._output_perm = \
            TemporalBasisTrafo._intermediate_and_output_shape_and_perms(
                input_shape, self.n_units, self.q, self.N)

    def call(self, xs):
        # Reshape the input into the shape required by tf.nn.convolution.
        # This will give us an array of shape (n_batch * n_units, N, 1), where
        # "n_batch" is a product of all "extra" dimensions.
        xs_transposed = tf.transpose(xs, perm=self._intermediate_perm)
        xs_reshaped = tf.reshape(xs_transposed, self._intermediate_shape)

        # Pad the second input dimension (corresponding to the input samples)
        # so we have 2 * N - 1 input samples.
        xs_padded = tf.pad(xs_reshaped, self.pad)

        # Compute the convolution with the basis transformation matrix H.
        ys = tf.nn.convolution(xs_padded, self.H)

        # Now reshape the output to the output dimensions we computed above and
        # permute the dimensions back so they match the desired input shape.
        ys_reshaped = tf.reshape(ys, self._output_shape)
        ys_transposed = tf.transpose(ys_reshaped, perm=self._output_perm)

        # Done!
        return ys_transposed

