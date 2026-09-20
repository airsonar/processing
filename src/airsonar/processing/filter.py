# SPDX-FileCopyrightText: AirSonar contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Filter design and application."""

from typing import Literal

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


def matched_filter(
    data: xr.DataArray,
    signal: xr.DataArray,
    method: Literal["fft"] = "fft",
    data_dim: str = "time",
    signal_dim: str = "time",
) -> xr.DataArray:
    """Apply a matched filter to some data.

    Parameters
    ----------
    data
        The data to apply the matched filter to.
    signal
        The signal to match.
    method
        The implementation method. Currently only `fft` is supported, which performs the
        filtering in the Fourier domain.
    data_dim
        The dimension of `data` to apply the filter along.
    signal_dim
        The dimension of `signal` that the signal is defined along.

    Returns
    -------
    matched : xr.DataArray
        An array with the broadcast dimensions of `data` and `signal` containing the
        matched-filtered data.

    """
    is_complex = data.dtype.kind == "c" or signal.dtype.kind == "c"

    # Assume that the signal and data will have different lengths along the dimension,
    # and rename the signal dimension to avoid coordinate mismatches.
    signal = signal.rename({signal_dim: "__signal_time__"})

    if method == "fft":
        return xr.apply_ufunc(
            _matched_filter_fft,
            data,
            signal,
            input_core_dims=[[data_dim], ["__signal_time__"]],
            output_core_dims=[[data_dim]],
            dask="parallelized",
            dask_gufunc_kwargs={
                "meta": np.empty(0, dtype=np.complex128 if is_complex else np.float64)
            },
            keep_attrs=True,
        )

    raise ValueError(f"unknown method '{method}'")


def _matched_filter_fft(data: np.ndarray, signal: np.ndarray):
    # To avoid circular effects, we will add a guard band by padding the end of the
    # arrays with zeros. Find the required final length. We make it even so both the
    # real and complex FFTs have the same padding.
    N = data.shape[-1] + signal.shape[-1] - 1
    if N % 2 == 1:
        N += 1

    # Decide on which FFT functions to use.
    is_complex = data.dtype.kind == "c" or signal.dtype.kind == "c"
    fft = np.fft.fft if is_complex else np.fft.rfft
    ifft = np.fft.ifft if is_complex else np.fft.irfft

    # Take the Fourier transforms. A trace with NaNs would have an all-NaN spectrum, so
    # we set those to zero. As N is larger than the length of either input, the FFT
    # routines will use zeros for the missing inputs thus performing our padding.
    D = fft(np.nan_to_num(data, copy=True, nan=0), n=N, axis=-1)
    S = fft(np.nan_to_num(signal, copy=True, nan=0), n=N, axis=-1)

    # Conjugate-multiply the spectra, take the inverse FFT and unpad.
    return ifft(D * S.conj(), axis=-1)[..., : data.shape[-1]]
