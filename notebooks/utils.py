#  Adaptive Filter Benchmark
#  Copyright (C) 2020 Andreas Stöckel
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import copy
import dataclasses
import numpy as np
import scipy.signal
import scipy.linalg


def nts(T, dt=1e-3):
    return int(T / dt + 1e-9)


def mkrng(rng=np.random):
    """
    Derives a new random number generator from the given random number
    generator.
    """
    return np.random.RandomState(rng.randint(1 << 31))


class FilteredGaussianSignal:
    """
    The FilteredGaussianSignal class generates a low-pass filtered white noise
    signal.
    """
    def __init__(self,
                 n_dim=1,
                 freq_low=None,
                 freq_high=1.0,
                 order=4,
                 dt=1e-3,
                 rng=np.random,
                 rms=0.5):
        assert (not freq_low is None) or (not freq_high is None)

        # Copy the given parameters
        self.n_dim = n_dim
        self.dt = dt
        self.rms = rms

        # Derive a new random number generator from the given rng. This ensures
        # that the signal will always be the same for a given random state,
        # independent of other
        self.rng = mkrng(rng)

        # Build the Butterworth filter
        if freq_low is None:
            btype = "lowpass"
            Wn = freq_high
        elif freq_high is None:
            btype = "highpass"
            Wn = freq_low
        else:
            btype = "bandpass"
            Wn = [freq_low, freq_high]
        self.b, self.a = scipy.signal.butter(N=order,
                                             Wn=Wn,
                                             btype=btype,
                                             analog=False,
                                             output='ba',
                                             fs=1.0 / dt)

        # Scale the output to reach the RMS
        self.b *= rms / np.sqrt(2.0 * dt * freq_high)

        # Initial state
        self.zi = np.zeros((max(len(self.a), len(self.b)) - 1, self.n_dim))

    def __call__(self, n_smpls):
        # Generate some random input
        xs = self.rng.randn(n_smpls, self.n_dim)

        # Filter each dimension independently, save the final state so multiple
        # calls to this function will create a seamless signal
        ys = np.empty((n_smpls, self.n_dim))
        for i in range(self.n_dim):
            ys[:, i], self.zi[:, i] = scipy.signal.lfilter(self.b,
                                                           self.a,
                                                           xs[:, i],
                                                           zi=self.zi[:, i])
        return ys


@dataclasses.dataclass
class EnvironmentDescriptor:
    """
    Dataclass containing run-time information about an instanciated environment.
    """
    n_state_dim: int = 1
    n_observation_dim: int = 1
    n_control_dim: int = 0


class Environment:
    def do_init(self):
        raise NotImplemented("do_init not implemented")

    def do_step(self, n_smpls):
        raise NotImplemented("do_step not implemented")

    def __init__(self, dt=1e-3, rng=np.random, *args, **kwargs):
        """
        Initializes the environment.

        dt: is the timestep.
        rng: is the random number generator.
        """
        assert dt > 0.0

        # Copy the given arguments
        self._dt = dt
        self._rng = mkrng(rng)

        self._descr = self.do_init(*args, **kwargs)

    def step(self, n_smpls):
        """
        Executes the environment for the specified number of samples. Returns
        the n_smpls x n_state_dim matrix containing the state, a
        n_smpls x n_observation_dim matrix of observations, and a n_smpls x
        n_control_dim matrix of control dimensions.
        """

        # Make sure the number of samples is non-negative
        assert int(n_smpls) >= 0

        # Call the actual implementation of step and destructure the return
        # value
        xs, zs, us = self.do_step(int(n_smpls))

        # Make sure the resulting arrays have the right dimensionality
        xs, zs, us = np.asarray(xs), np.asarray(zs), np.asarray(us)
        if xs.ndim != 2:
            xs = xs.reshape(n_smpls, -1)
        if zs.ndim != 2:
            zs = zs.reshape(n_smpls, -1)
        if us.ndim != 2:
            us = us.reshape(n_smpls, -1)

        # Make sure the returned arrays have the right shape
        assert xs.shape == (n_smpls, self.n_state_dim)
        assert zs.shape == (n_smpls, self.n_observation_dim)
        assert us.shape == (n_smpls, self.n_control_dim)

        return xs, zs, us

    def clone(self):
        """
        Produces a copy of this Environment instance that will behave exactly
        as this one, but is decoupled from this instance.
        """
        return copy.deepcopy(self)

    @property
    def dt(self):
        return self._dt

    @property
    def rng(self):
        return self._rng

    @property
    def descr(self):
        return self._descr

    @property
    def n_state_dim(self):
        return self._descr.n_state_dim

    @property
    def n_observation_dim(self):
        return self._descr.n_observation_dim

    @property
    def n_control_dim(self):
        return self._descr.n_control_dim


class EnvironmentWithSignalBase(Environment):
    def do_init(self, signal_kwargs=None):
        # Assemble the parameters that are being passed to the 1D signal
        # generator
        if signal_kwargs is None:
            signal_kwargs = {}
        if not "freq_high" in signal_kwargs:
            signal_kwargs["freq_high"] = 0.1
        if not "rms" in signal_kwargs:
            signal_kwargs["rms"] = 1.0

        # Initialize the filtered signal instance
        self.signal = FilteredGaussianSignal(n_dim=1,
                                             dt=self.dt,
                                             rng=self.rng,
                                             **signal_kwargs)


class FrequencyModulatedSine(EnvironmentWithSignalBase):
    def do_init(self,
                f0=0.0,
                f1=2.0,
                signal_kwargs=None,
                use_control_dim=True):
        # Call the inherited constructor
        super().do_init(signal_kwargs=signal_kwargs)

        # Copy the given arguments
        self.f0 = f0
        self.f1 = f1
        self.use_control_dim = use_control_dim

        # Initialize the current state
        self.phi = 0.0

        return EnvironmentDescriptor(1, 1, 1 if use_control_dim else 0)


    def do_step(self, n_smpls):
        # Compute the frequencies
        us = 0.5 * self.signal(n_smpls)
        fs = (self.f1 - self.f0) * 0.5 * (us + 1.0) + self.f0

        # Integrate the frequencies to obtain the phases
        phis = self.phi + np.cumsum(fs) * (2.0 * np.pi * self.dt)
        xs = phis % (2.0 * np.pi)
        self.phi = xs[-1]

        # Compute the observation
        zs = np.sin(xs)

        # Do not return the control data if "use_control_dim" is set to false
        if not self.use_control_dim:
            us = np.zeros((n_smpls, 0))

        return xs, zs, us

