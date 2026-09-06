# SPDX-FileCopyrightText: AirSonar contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Conversion of PDM signals to pressure signals."""

import numpy as np
from scipy import signal
import xarray as xr


def binary_to_float_nrz(
    data: xr.DataArray, high_value: float = 0.5, invert: bool = False
) -> xr.DataArray:
    """Convert a binary signal to floating-point non-return-to-zero.

    When the raw PDM signal from the microphone is recorded, it may be biased away from
    zero (e.g., it may be stored as ones and zeros). This can cause issues when
    converting it to a pressure signal.

    This function takes any binary signal and converts it to a non-return-to-zero signal
    stored as floats. If the given `high_value` parameter is denoted as H, then high
    values in the input will correspond to +H and low values in the output to -H.

    Parameters
    ----------
    data
        The binary data to convert. This can use any values to represent high and low,
        as long as there are exactly two unique values in the array.
    high_value
        The value of a high bit in the output. The low bits will have the negated value.
    invert
        If True, the output data is inverted with respect to the input (high becomes low
        and vice-versa). This can be used to convert PDM signals where an increasing
        density of high bits corresponds to a decreasing pressure.

    Returns
    -------
    nrz : xr.DataArray
        The binary data as an array of floats centred about zero.

    """
    # Check we have binary data, and get the current levels.
    levels = np.unique_counts(data).values
    if len(levels) != 2:
        raise ValueError("conversion only possible with binary data")

    # Calculate and apply the appropriate offset and scaling.
    offset = levels.mean()
    scale = 2 * high_value / np.abs(levels[1] - levels[0])
    if invert:
        scale *= -1

    return scale * (data.astype(float) - offset)


def downsample_integer(
    data: xr.DataArray, stages: list[int], dim: str = "time"
) -> xr.DataArray:
    """Downsample a signal by integer factors in stages.

    This uses the [scipy.signal.decimate][] function to perform each decimation. A FIR
    anti-aliasing filter is used with an order of 20 times the factor of the stage. The
    zero-phase option is set to avoid phase shifts when applying the filter.

    Parameters
    ----------
    data
        The data to downsample.
    stages
        A list of integer factors to downsample. A value [8, 6] downsamples first by a
        factor of 8 and then by a factor of 6.
    dim
        The dimension to downsample along. The coordinates are assumed to be equally
        spaced along this dimension.

    Returns
    -------
    downsampled : xr.DataArray
        The downsampled data with the same dimensions and attributes as the input data.

    """
    # scipy.signal.decimate() upcasts integers and float16. Perform the check with the
    # same logic to ensure we tell apply_ufunc() the correct type.
    if not np.issubdtype(data.dtype, np.inexact) or data.dtype == np.float16:
        output_dtype = np.dtype("f8")
    else:
        output_dtype = data.dtype

    for factor in stages:
        # Subsample the coordinates.
        new_coords = data[dim].values[::factor]

        # And decimate the data.
        data = xr.apply_ufunc(
            signal.decimate,
            data,
            kwargs={"q": factor, "ftype": "fir", "zero_phase": True, "axis": -1},
            input_core_dims=[[dim]],
            output_core_dims=[[dim]],
            exclude_dims={dim},
            keep_attrs=True,
            dask="parallelized",
            output_dtypes=[output_dtype],
            dask_gufunc_kwargs={
                "output_sizes": {dim: len(new_coords)},
            },
        )
        data = data.assign_coords({dim: new_coords})

    return data
