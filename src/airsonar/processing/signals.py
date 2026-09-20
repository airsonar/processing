# SPDX-FileCopyrightText: AirSonar contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Signal generation."""

import numpy as np
from numpy.typing import ArrayLike
import xarray as xr


def lfm(
    time: ArrayLike,
    f_start: float,
    f_stop: float,
    duration: float,
    invalid: float = 0.0,
) -> xr.DataArray:
    r"""Generate samples of a linear frequency modulated chirp.

    The instantaneous frequency of the chirp varies linearly from the start frequency to
    the stop frequency over the duration of the chirp. The value of the signal is given
    by

    $$
    s(t) = \begin{cases}
        \sin(2\pi f_a t + \pi K t^2)  & t \leq 0 \leq T, \\
                        v               & \text{otherwise},
    \end{cases}
    $$

    where $f_a$ is the start frequency, $K = (f_b - f_a)/T$ is the chirp rate, $f_b$ is
    the stop frequency, $T$ is the duration and $v$ is the invalid value passed to this
    function. The instantaneous frequency is

    $$
    f(t) = \begin{cases}
        f_a + Kt  & t \leq0 \leq T, \\
            0     & \text{otherwise}.
    \end{cases}
    $$

    Parameters
    ----------
    time
        The times in seconds since the start of the chirp to generate samples for.
    f_start
        The start frequency of the chirp.
    f_stop
        The stop frequency of the chirp.
    duration
        The duration of the chirp in seconds.
    invalid
        The value to use in the output for times before or after the chirp.

    Returns
    -------
    chirp : xr.DataArray
        The samples of the chirp at the requested times. If a non-DataArray value was
        given for the `time` input, the output DataArray will have a dimension named
        `time` with the input times as coordinates.

    """
    # Wrap time into a DataArray.
    if not isinstance(time, xr.DataArray):
        time = np.atleast_1d(time)
        if time.ndim != 1:
            raise ValueError("multi-dimensional times must be given as a DataArray")

        time = xr.DataArray(time, coords=[("time", time)])

    # Evaluate for all times and then clip to the duration.
    K = (f_stop - f_start) / duration
    signal: xr.DataArray = np.sin((2 * np.pi * f_start * time) + (np.pi * K * time**2))  # type:ignore[assignment]
    return signal.where((time >= 0) & (time <= duration), other=invalid)


def lpm(
    time: ArrayLike,
    f_start: float,
    f_stop: float,
    duration: float,
    invalid: float = 0.0,
) -> xr.DataArray:
    r"""Generate samples of a linear period modulated chirp.

    The instantaneous period of the chirp varies linearly over the duration of the
    chirp. This is the result of the instantaneous frequency varying hyperbolically from
    the start frequency to the stop frequency over the duration of the chirp (this
    signal is sometimes known as a *hyperbolic frequency modulated chirp*. In effect,
    the signal spends a larger fraction of the chirp duration at lower frequencies.

    The parameter $\tau_0$ is defined as

    $$
    \tau_0 = \frac{f_b}{f_b - f_a} T,
    $$

    where $f_a$ is the start frequency, $f_b$ is the stop frequency and $T$ is the
    duration. The starting phase is then $\phi_0 = 2\pi f_a \tau_0$. The value of the
    signal is given by

    $$
    s(t) = \begin{cases}
        \sin\left(\phi_0 \ln\dfrac{\tau_0 - t}{\tau_0}\right)  & t \leq 0 \leq T, \\
                        v               & \text{otherwise},
    \end{cases}
    $$


    where $v$ is the invalid value passed to this function. The instantaneous frequency
    is

    $$
    f(t) = \begin{cases}
        \dfrac{f_a \tau_0}{\tau_0 - t}  & t \leq0 \leq T, \\
                        0               & \text{otherwise}.
    \end{cases}
    $$

    Parameters
    ----------
    time
        The times in seconds since the start of the chirp to generate samples for.
    f_start
        The start frequency of the chirp.
    f_stop
        The stop frequency of the chirp.
    duration
        The duration of the chirp in seconds.
    invalid
        The value to use in the output for times before or after the chirp.

    Returns
    -------
    chirp : xr.DataArray
        The samples of the chirp at the requested times. If a non-DataArray value was
        given for the `time` input, the output DataArray will have a dimension named
        `time` with the input times as coordinates.

    """
    # Wrap time into a DataArray.
    if not isinstance(time, xr.DataArray):
        time = np.atleast_1d(time)
        if time.ndim != 1:
            raise ValueError("multi-dimensional times must be given as a DataArray")

        time = xr.DataArray(time, coords=[("time", time)])

    # Evaluate for all times and then clip to the duration.
    tau0 = (f_stop / (f_stop - f_start)) * duration
    phi0 = 2 * np.pi * f_start * tau0
    signal: xr.DataArray = np.sin(phi0 * np.log((tau0 - time) / tau0))  # type:ignore[assignment]
    return signal.where((time >= 0) & (time <= duration), other=invalid)
