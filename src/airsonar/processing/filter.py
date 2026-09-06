# SPDX-FileCopyrightText: AirSonar contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Filter design and application."""

import numpy as np
from scipy import signal
import xarray as xr


def fir_least_squares(
    data: xr.DataArray,
    low_edge: float | None,
    high_edge: float | None,
    transition_width: float,
    taps: int = 151,
    dim: str = "time",
) -> xr.DataArray:
    """Apply a least-squares optimised FIR filter to some data.

    This uses [scipy.signal.firls][] to calculate the coefficients for each tap. The
    filter is applied using [scipy.signal.filtfilt][] which applies the filter both
    forward and backwards to avoid phase shifts from the filtering. Note that this
    results in a filter order of twice the original design.

    Parameters
    ----------
    data
        The data to apply the filter to.
    low_edge
        The low edge of the passband in Hertz. If None, a low-pass filter is generated.
    high_edge
        The high edge of the passband in Hertz. If None, a high-pass filter is
        generated. Cannot be None if `low_edge` is None.
    transition_width
        The desired transition width at each edge in Hertz. The ideal filter would be
        zero below (low_edge - transition_width) and above (high_edge +
        transition_width).
    taps
        The number of taps in the filter. This must be positive and odd.
    dim
        The dimension to filter along. The coordinates are assumed to be equally spaced
        along this dimension.

    Returns
    -------
    filtered : xr.DataArray
        The filtered data. This will have the same coordinates as the input data.

    """
    if transition_width <= 0:
        raise ValueError("transition_width must be greater than zero")

    # Compute the sampling rate, assuming the coordinates are constantly spaced.
    fs = 1 / float(data[dim][1] - data[dim][0])
    nyquist = fs / 2

    # Low-pass filter.
    if low_edge is None:
        if high_edge is None:
            raise ValueError("only one of low_edge and high_edge can be None")
        if (high_edge + transition_width) > nyquist:
            raise ValueError(
                "high_edge + transition_width is higher than the Nyquist frequency"
            )

        bands = [0, high_edge, high_edge + transition_width, nyquist]
        desired = [1, 1, 0, 0]

    # High-pass filter.
    elif high_edge is None:
        if low_edge < transition_width:
            raise ValueError("low_edge cannot be smaller than transition_width")

        bands = [0, low_edge - transition_width, low_edge, nyquist]
        desired = [0, 0, 1, 1]

    # Bandpass filter.
    else:
        if low_edge < transition_width:
            raise ValueError("low_edge cannot be smaller than transition_width")
        if (high_edge + transition_width) > nyquist:
            raise ValueError(
                "high_edge + transition_width is higher than the Nyquist frequency"
            )

        bands = [
            0,
            low_edge - transition_width,
            low_edge,
            high_edge,
            high_edge + transition_width,
            nyquist,
        ]
        desired = [0, 0, 1, 1, 0, 0]

    # And generate.
    coeffs = signal.firls(taps, bands=bands, desired=desired, fs=fs)

    # Create a ufunc to apply the filter.
    def apply_fir_least_squares(values: np.ndarray) -> np.ndarray:
        return signal.filtfilt(coeffs, [1], values, axis=-1)

    # And apply that ufunc.
    filtered = xr.apply_ufunc(
        apply_fir_least_squares,
        data,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        keep_attrs=True,
        dask="parallelized",
        dask_gufunc_kwargs={"meta": np.empty(0, dtype=float)},
    )
    return filtered
