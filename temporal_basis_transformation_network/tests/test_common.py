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

from temporal_basis_transformation_network.common import *


def test_compute_shapes_and_permutations_forward():
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
        def f(S, n_units, q, N):
            return compute_shapes_and_permutations(S, n_units, q, N, pad, True,
                                                   Forward)

        # Iterate over multiple values for M_in
        for M_in in [1, N, N * 2]:
            M_out = (M_in if pad else max(1, M_in - N + 1))

            # Single unit
            assert f((M_in, 1), 1, q, N) == \
                    ((M_in, 1),                  # input_shape_pre
                     (1, 0),                     # input_perm
                     (1, M_in, 1),               # input_shape_post
                     (1, M_out, q),              # output_shape_pre
                     (1, 0, 2),                  # output_perm
                     (M_out, q))                 # output_shape_post

            assert f((2, M_in, 1), 1, q, N) == \
                    ((2, M_in, 1),               # input_shape_pre
                     (0, 2, 1),                  # input_perm
                     (2, M_in, 1),               # input_shape_post
                     (2, 1, M_out, q),           # output_shape_pre
                     (0, 2, 1, 3),               # output_perm
                     (2, M_out, q))              # output_shape_post

            assert f((5, 2, M_in, 1), 1, q, N) == \
                    ((5, 2, M_in, 1),            # input_shape_pre
                     (0, 1, 3, 2),               # input_perm
                     (10, M_in, 1),              # input_shape_post
                     (5, 2, 1, M_out, q),        # output_shape_pre
                     (0, 1, 3, 2, 4),            # output_perm
                     (5, 2, M_out, q))           # output_shape_post

            # Multiple units
            assert f((M_in, 7), 7, q, N) == \
                    ((M_in, 7),                  # input_shape_pre
                     (1, 0),                     # input_perm
                     (7, M_in, 1),               # input_shape_post
                     (7, M_out, q),              # output_shape_pre
                     (1, 0, 2),                  # output_perm
                     (M_out, 7 * q))             # output_shape_post

            assert f((2, M_in, 7), 7, q, N) == \
                    ((2, M_in, 7),               # input_shape_pre
                     (0, 2, 1),                  # input_perm
                     (14, M_in, 1),              # input_shape_post
                     (2, 7, M_out, q),           # output_shape_pre
                     (0, 2, 1, 3),               # output_perm
                     (2, M_out, 7 * q))          # output_shape_post

            assert f((5, 2, M_in, 7), 7, q, N) == \
                    ((5, 2, M_in, 7),            # input_shape_pre
                     (0, 1, 3, 2),               # input_perm
                     (70, M_in, 1),              # input_shape_post
                     (5, 2, 7, M_out, q),        # output_shape_pre
                     (0, 1, 3, 2, 4),            # output_perm
                     (5, 2, M_out, 7 * q))       # output_shape_post

            # "None" in the shape descriptor acts as a wildcard.
            assert f((None, M_in, 1), 1, q, N) == \
                    ((-1, M_in, 1),              # input_shape_pre
                     (0, 2, 1),                  # input_perm
                     (-1, M_in, 1),              # input_shape_post
                     (-1, 1, M_out, q),          # output_shape_pre
                     (0, 2, 1, 3),               # output_perm
                     (-1, M_out, q))             # output_shape_post

            assert f((None, 5, M_in, 1), 1, q, N) == \
                    ((-1, 5, M_in, 1),           # input_shape_pre
                     (0, 1, 3, 2),               # input_perm
                     (-1, M_in, 1),              # input_shape_post
                     (-1, 5, 1, M_out, q),       # output_shape_pre
                     (0, 1, 3, 2, 4),            # output_perm
                     (-1, 5, M_out, q))          # output_shape_post


def test_compute_shapes_and_permutations_inverse():
    q, N = 6, 100
    # Create a short-hand for the function under test
    def f(S, n_units, q, N):
        return compute_shapes_and_permutations(S, n_units, q, N, True, True,
                                               Inverse)

    # Iterate over multiple values for M_in
    for M_in in [1, N, N * 2]:
        M_out = M_in

        # Single unit
        assert f((M_in, q), 1, q, N) == \
                ((M_in, 1, q),                   # input_shape_pre
                 (1, 0, 2),                      # input_perm
                 (1, M_in, q),                   # input_shape_post
                 (1, M_out, N),                  # output_shape_pre
                 (1, 0, 2),                      # output_perm
                 (M_out, N))                     # output_shape_post

        assert f((2, M_in, q), 1, q, N) == \
                ((2, M_in, 1, q),            # input_shape_pre
                 (0, 2, 1, 3),               # input_perm
                 (2, M_in, q),               # input_shape_post
                 (2, 1, M_out, N),           # output_shape_pre
                 (0, 2, 1, 3),               # output_perm
                 (2, M_out, N))              # output_shape_post

        assert f((5, 2, M_in, q), 1, q, N) == \
                ((5, 2, M_in, 1, q),         # input_shape_pre
                 (0, 1, 3, 2, 4),            # input_perm
                 (10, M_in, q),              # input_shape_post
                 (5, 2, 1, M_out, N),        # output_shape_pre
                 (0, 1, 3, 2, 4),            # output_perm
                 (5, 2, M_out, N))           # output_shape_post

        # Multiple units
        assert f((M_in, 7 * q), 7, q, N) == \
                ((M_in, 7, q),               # input_shape_pre
                 (1, 0, 2),                  # input_perm
                 (7, M_in, q),               # input_shape_post
                 (7, M_out, N),              # output_shape_pre
                 (1, 0, 2),                  # output_perm
                 (M_out, 7 * N))             # output_shape_post

        assert f((2, M_in, 7 * q), 7, q, N) == \
                ((2, M_in, 7, q),            # input_shape_pre
                 (0, 2, 1, 3),               # input_perm
                 (14, M_in, q),              # input_shape_post
                 (2, 7, M_out, N),           # output_shape_pre
                 (0, 2, 1, 3),               # output_perm
                 (2, M_out, 7 * N))          # output_shape_post

        assert f((5, 2, M_in, 7 * q), 7, q, N) == \
                ((5, 2, M_in, 7, q),         # input_shape_pre
                 (0, 1, 3, 2, 4),            # input_perm
                 (70, M_in, q),              # input_shape_post
                 (5, 2, 7, M_out, N),        # output_shape_pre
                 (0, 1, 3, 2, 4),            # output_perm
                 (5, 2, M_out, 7 * N))       # output_shape_post

        # "None" in the shape descriptor acts as a wildcard.
        assert f((None, M_in, q), 1, q, N) == \
                ((-1, M_in, 1, q),           # input_shape_pre
                 (0, 2, 1, 3),               # input_perm
                 (-1, M_in, q),              # input_shape_post
                 (-1, 1, M_out, N),          # output_shape_pre
                 (0, 2, 1, 3),               # output_perm
                 (-1, M_out, N))             # output_shape_post

        assert f((None, 5, M_in, q), 1, q, N) == \
                ((-1, 5, M_in, 1, q),        # input_shape_pre
                 (0, 1, 3, 2, 4),            # input_perm
                 (-1, M_in, q),              # input_shape_post
                 (-1, 5, 1, M_out, N),       # output_shape_pre
                 (0, 1, 3, 2, 4),            # output_perm
                 (-1, 5, M_out, N))          # output_shape_post

