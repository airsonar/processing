# SPDX-FileCopyrightText: AirSonar contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Image reconstruction."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import hilbert
import xarray as xr


def _ensure_1d(values: ArrayLike, default_dim: str) -> xr.DataArray:
    """Check a coordinate variable is 1D and normalise to a DataArray.

    Parameters
    ----------
    values
        Values for the coordinates.
    default_dim
        The name to use for the dimension if `values` is not a DataArray.

    Returns
    -------
    xr.DataArray

    """
    if not isinstance(values, xr.DataArray):
        values = np.atleast_1d(values)
        if values.ndim != 1:
            raise ValueError(f"values for {default_dim} must be one-dimensional")

        return xr.DataArray(values, coords=[(default_dim, values)])

    if values.ndim != 1:
        raise ValueError(f"values for {default_dim} must be one-dimensional")

    return values


def backproject(
    data: xr.DataArray,
    tx_position: xr.DataArray,
    rx_position: xr.DataArray,
    image_x: ArrayLike,
    image_y: ArrayLike,
    image_z: ArrayLike = 0,
    c: float = 343.0,
    rvg: bool = True,
    complex_image: bool = True,
    channel_dim: str = "channel",
    time_dim: str = "time",
    position_dim: str = "xyz",
) -> xr.DataArray:
    """Reconstruct an image using the backprojection algorithm.

    The backprojection algorithm takes each trace of data and projects it back onto the
    positions on the reconstruction surface that it could have originated from. For each
    trace, this results in an arc of data. When summed over all pings and channels, the
    data constructively adds in true target locations and destructively cancels
    elsewhere.

    Parameters
    ----------
    data
        The data to backproject.
    tx_position
        The position of the transmitter for each trace in `data`.
    rx_position
        The position of each receiver for each trace in `data`.
    image_x
        The x coordinates of the focal points to reconstruct at. This must be
        one-dimensional.
    image_y
        The y coordinates of the focal points to reconstruct at. This must be
        one-dimensional.
    image_z
        The z coordinates of the focal points to reconstruct at. This may be either
        scalar if z is constant for all focal points, or a two-dimensional array giving
        a value for each focal point.
    c
        The speed of sound to use during reconstruction.
    rvg
        Whether to apply a range-varying gain during reconstruction to compensate for
        spherical spreading losses. Note that backprojection has an inherent one-way
        gain. Setting this parameter to True applies the remaining one-way gain to
        complete this compensation. Setting it to False does not apply the remaining
        gain, leaving the image with its inherent compensation.
    complex_image
        If True and `data` is real, use [scipy.signal.hilbert][] to convert it to an
        analytic signal before backprojecting. If False, use the `data` array as given.
    channel_dim
        The dimension of `data` and `rx_position` corresponding to the receiver
        channels.
    time_dim
        The dimension of `data` corresponding to the sampling times.
    position_dim
        The dimension of `tx_position` and `rx_position` corresponding to the position
        coordinates.

    Returns
    -------
    image : xr.DataArray
        The complex image formed at the given focal points. This will be two-dimensional
        with the requested x and y coordinates.

    """
    # Check and convert coordinates to DataArrays.
    image_x = _ensure_1d(image_x, "x")
    image_y = _ensure_1d(image_y, "y")

    # Coerce z positions to either a float or a 2D DataArray.
    if not isinstance(image_z, xr.DataArray):
        image_z_arr = np.asarray(image_z)
        if image_z_arr.ndim == 2:
            image_z = xr.DataArray(image_z_arr, coords=[image_x, image_y])
        elif image_z_arr.ndim != 0:
            raise ValueError("image_z must be a float or a 2D array")

    # Decide on the output datatype based on the input and the complex_image flag.
    if data.dtype.kind == "c":
        output_dtype = data.dtype
    elif complex_image:
        output_dtype = complex  # type:ignore[assignment]
    else:
        output_dtype = data.dtype

    # Call the low-level ufunc on each block.
    image = xr.apply_ufunc(
        _backproject,
        data,
        tx_position,
        rx_position,
        input_core_dims=[
            [channel_dim, time_dim],
            [position_dim],
            [channel_dim, position_dim],
        ],
        kwargs={
            "image_x": image_x.values,
            "image_y": image_y.values,
            "image_z": image_z.values if isinstance(image_z, xr.DataArray) else image_z,
            "c": c,
            "t": data[time_dim].values,
            "complex_image": complex_image,
            "rvg": rvg,
        },
        output_core_dims=[["__image_x__", "__image_y__"]],
        dask="parallelized",
        dask_gufunc_kwargs={
            "meta": [np.empty(0, dtype=output_dtype)],
            "output_sizes": {"__image_x__": len(image_x), "__image_y__": len(image_y)},
        },
        keep_attrs=True,
    )

    # Sum over non-image dimensions.
    if image.ndim > 2:
        image = image.sum(image.dims[:-2])

    # And assign the dimensions and coordinates of the focal points.
    image = image.assign_coords(
        __image_x__=image_x.values,
        __image_y__=image_y.values,
    ).rename(
        __image_x__=image_x.dims[0],
        __image_y__=image_y.dims[0],
    )

    return image


def _backproject(
    data: np.ndarray,
    tx_position: np.ndarray,
    rx_position: np.ndarray,
    image_x: np.ndarray,
    image_y: np.ndarray,
    image_z: float | np.ndarray,
    c: float,
    rvg: bool,
    complex_image: bool,
    t: np.ndarray,
) -> np.ndarray:
    """Universal function implementing backprojection.

    This is called by the `backproject` function to backproject each block of data.

    Parameters
    ----------
    data
        The data to be backprojected as a (..., Nc, Nt) array.
    tx_position
        The position of the transmitter as a (..., xyz) array.
    rx_position
        The position of the receivers as a (..., Nc, xyz) array.
    image_x
        The x position of the focal points as a (Nx) array.
    image_y
        The y position of the focal points as a (Ny) array.
    image_z
        The z position of the focal points as either a float or an (Nx, Ny) array.
    c
        The speed of sound to use during backprojection.
    rvg
        Whether to complete the range-varying gain.
    complex_image
        If True and `data` is real, convert to an analytic signal with
        [scipy.signal.hilbert][].
    t
        The data sample times as a (Nt) array.

    Returns
    -------
    reconstructed : np.ndarray
        The backprojected data as an (..., Nx, Ny) array.

    """
    # Determine the relevant sizes.
    extra = data.shape[:-2]
    Nx = len(image_x)
    Ny = len(image_y)

    # Convert to an analytic signal if required.
    if data.dtype.kind != "c" and complex_image:
        data = hilbert(data)

    # Initialise the output.
    image = np.zeros(extra + (Nx, Ny), dtype=data.dtype)

    # Expand the x and y coordinates to a grid.
    x, y = np.meshgrid(image_x, image_y, indexing="ij")

    # Expand the z coordinates if required.
    z = np.asarray(image_z)
    if z.ndim == 0:
        z = np.full_like(x, image_z)

    # And stack into one array.
    focal = np.stack([x, y, z], -1)

    # And loop over all traces.
    for idx in np.ndindex(data.shape[:-1]):
        # Ranges and from that the two-way travel time. Note that idx includes a channel
        # index which we have to remove when extracting the tx position, and that the
        # resulting rx_position will have a size-1 channel dimension which we squeeze.
        r_tx = np.sqrt(np.sum((focal - tx_position[idx[:-1]]) ** 2, axis=-1))
        r_rx = np.sqrt(np.sum((focal - rx_position[idx].squeeze()) ** 2, axis=-1))
        img_t = (r_tx + r_rx) / c

        # Interpolate the data and sum.
        img_pc = np.interp(img_t, t, data[idx])
        if rvg:
            img_pc *= np.sqrt(r_tx * r_rx)
        image += img_pc

    return image
