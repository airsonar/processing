# SPDX-FileCopyrightText: AirSonar contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Conversion of PDM signals to pressure signals."""

import numpy as np
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
