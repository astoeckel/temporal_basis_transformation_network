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

import itertools
import numpy as np
import dlop_ldn_function_bases as bases
from temporal_basis_transformation_network import TemporalBasisTrafo


def _iterative_reference_implementation(q, xs):
    """
    Reference implementation feeding an input signal "xs" through a Legendre
    Delay Network.
    """
    assert xs.ndim == 1
    N, = xs.shape
    A, B = bases.discretize_lti(1.0 / N, *bases.mk_ldn_lti(q))
    res, m = np.zeros((N, q)), np.zeros(q)
    for i in range(N):
        m = A @ m + B * xs[i]
        res[i] = m
    return np.asarray(res, dtype=np.float32)


def test_intermediate_and_output_shape_and_perms_pad_input():
    """
    Makes sure the the internal "_compute_intermediate_and_output_shape"
    function works correctly in the case of pad=True and the number of input
    samples being equal to the filter size.  This function is responsible for
    making sure that many different input shape configurations are conveniently
    supported by the network layer.
    """

    q, N = 6, 100
    for pad in [True, False]:
        # Create a short-hand for the function under test
        f = lambda S, n, q, N: \
            TemporalBasisTrafo._intermediate_and_output_shape_and_perms(
                S, n, q, N, pad=pad)

        # Iterate over multiple values for M_in
        for M_in in [1, N, N * 2]:
            M_out = (M_in if pad else max(1, M_in - N + 1))

            # Special cases for one input unit
            assert f((M_in, 1), 1, q, N) == \
                    ((1, M_in, 1), (1, 0),
                     (1, M_out, q), (1, 0, 2))
            assert f((2, M_in, 1), 1, q, N) == \
                    ((2, M_in, 1), (0, 2, 1),
                     (2, 1, M_out, q), (0, 2, 1, 3))
            assert f((5, 2, M_in, 1), 1, 6, N) == \
                    ((10, M_in, 1), (0, 1, 3, 2),
                     (5, 2, 1, M_out, q), (0, 1, 3, 2, 4))

            # Multiple units
            assert f((M_in, 7), 7, q, N) == \
                    ((7, M_in, 1), (1, 0), \
                     (7, M_out, q), (1, 0, 2))
            assert f((2, M_in, 7), 7, q, N) == \
                    ((14, M_in, 1), (0, 2, 1), \
                     (2, 7, M_out, q), (0, 2, 1, 3))
            assert f((5, 2, M_in, 7), 7, q, N) == \
                    ((70, M_in, 1), (0, 1, 3, 2), \
                     (5, 2, 7, M_out, q), (0, 1, 3, 2, 4))

            # "None" in the shape descriptor acts as a wildcard.
            assert f((None, M_in, 1), 1, q, N) == \
                    ((-1, M_in, 1), (0, 2, 1), \
                     (-1, 1, M_out, q), (0, 2, 1, 3))
            assert f((None, 5, M_in, 1), 1, q, N) == \
                    ((-1, M_in, 1), (0, 1, 3, 2), \
                     (-1, 5, 1, M_out, q), (0, 1, 3, 2, 4))


def _test_impulse_response_generic(q, N, dims_pre=tuple(), dims_post=(1,)):
    """
    Generates some test data of the shape (*dims_pre, N, *dims_post) and
    feeds it into a TemporalBasisTrafo instances with a Legendre Delay Network
    basis. Computes the reference Delay Network output and compares the output
    to the output returned by the TemporalBasisTrafo.
    """

    # Generate some test data
    q, N = 10, 100
    rng = np.random.RandomState(49818)
    xs = rng.randn(*dims_pre, N, *dims_post)

    # Generate the reference output; iterate over all batch dimensions
    H = bases.mk_ldn_basis(q, N, normalize=False)
    ys_ref = np.zeros((*dims_pre, N, *dims_post, q))
    dims_cat = tuple((*dims_pre, *dims_post))
    for idcs in itertools.product(*map(range, dims_cat)):
        # Split the indices into the pre and post indices
        idcs_pre = idcs[:len(dims_pre)]
        idcs_post = idcs[len(dims_pre):]

        # Assemble the source/target slice
        sel = tuple((*idcs_pre, slice(None), *idcs_post))
        ys_ref[sel] = _iterative_reference_implementation(q, xs[sel])

    # Create the tensorflow network with the LDN basis
    ys = TemporalBasisTrafo(
        H, dims_post[0] if len(dims_post) > 0 else 1)(xs).numpy()
    assert ys.shape == ys_ref.shape

    # Make sure the absolute error is smaller than 1e-6
    np.testing.assert_allclose(ys, ys_ref, atol=1e-6)


def test_impulse_response_single_batch_single_unit():
    # Most simple case. Single unit, 100 input samples
    _test_impulse_response_generic(10, 100)

def test_impulse_response_multiple_batches_single_unit():
    # Same as above, but add some arbitrary batch dimensions
    _test_impulse_response_generic(10, 100, (1, 1))
    _test_impulse_response_generic(10, 100, (5, 1))
    _test_impulse_response_generic(10, 100, (5, 3, 1))
    _test_impulse_response_generic(10, 100, (5, 2, 3, 1))

def test_impulse_response_single_batch_multiple_units():
    # Single batch dimension; seven individual units
    _test_impulse_response_generic(10, 100, tuple(), (7,))


def test_impulse_response_multiple_batches_multiple_unit():
    # Arbitrary batch dimensions but seven individual units
    _test_impulse_response_generic(10, 100, (1,), (7,))
    _test_impulse_response_generic(10, 100, (5,), (7,))
    _test_impulse_response_generic(10, 100, (5, 3), (7,))
    _test_impulse_response_generic(10, 100, (5, 2, 3), (7,))

