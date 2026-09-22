# SPDX-FileCopyrightText: AirSonar contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import dask
import numpy as np
from numpy.typing import ArrayLike
import pytest
from scipy.signal import hilbert
import xarray as xr

from airsonar.processing import reconstruct


@pytest.fixture(scope="module")
def constant_z_data():
    # Ping angles.
    theta = np.arange(0, 2 * np.pi, np.pi / 20)

    # Transmit positions.
    tx_x = np.cos(theta)
    tx_y = np.sin(theta)
    tx_z = np.full_like(tx_x, 0.7)
    tx_pos = xr.DataArray(
        [tx_x, tx_y, tx_z],
        coords=[("xyz", ["x", "y", "z"]), ("ping", np.arange(len(tx_x)))],
    )
    tx_pos = tx_pos.transpose(..., "xyz")

    # Channel spacing and relative receiver positions.
    d_rx = 8.5e-3
    rx_relx = (np.arange(16) - 7.5) * d_rx
    rx_x = tx_x + rx_relx[:, None] * np.cos(theta - np.pi / 2)
    rx_y = tx_y + rx_relx[:, None] * np.sin(theta - np.pi / 2)
    rx_z = np.full_like(rx_x, 0.72)
    rx_pos = xr.DataArray(
        [rx_x, rx_y, rx_z],
        coords=[tx_pos["xyz"], ("channel", np.arange(16)), tx_pos["ping"]],
    )
    rx_pos = rx_pos.transpose(..., "xyz")

    # Sample times and sound speed
    fs = 100e3
    t = np.arange(0, 10e-3, 1 / fs)
    c = 343.0

    # Two-way ranges to target.
    tgt = np.array([10e-2, 4e-2, 0])
    r_tx = np.sqrt(((tx_pos - tgt) ** 2).sum("xyz"))
    r_rx = np.sqrt(((rx_pos - tgt) ** 2).sum("xyz"))

    # Corresponding travel time.
    twtt = (r_tx + r_rx) / c

    # Apply the corresponding phase shifts to the Fourier transform of an impulse.
    _f = np.fft.rfftfreq(len(t), t[1] - t[0])
    f = xr.DataArray(_f, coords=[("f", _f)])
    S = np.exp(-2j * np.pi * f * twtt)

    # And inverse transform to get our matched-filtered data.
    data = xr.apply_ufunc(
        np.fft.irfft, S, input_core_dims=[["f"]], output_core_dims=[["time"]]
    )
    data = data.assign_coords(time=t)

    return data, tx_pos, rx_pos, tgt, c, fs


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize(
    "real_data,complex_image", [[True, True], [True, False], [False, False]]
)
def test_backproject_constant_z(
    constant_z_data, chunked: bool, real_data: bool, complex_image: bool
):
    data, tx_pos, rx_pos, tgt, c, fs = constant_z_data

    if not real_data:
        data = xr.apply_ufunc(
            hilbert, data, input_core_dims=[["time"]], output_core_dims=[["time"]]
        )

    # Common reconstruction arguments to form a 1mm resolution image in a 10cm by 10cm
    # square around the target position.
    args = [
        tx_pos,
        rx_pos,
        np.arange(tgt[0] - 5e-2, tgt[0] + 5e-2, 1e-3),
        np.arange(tgt[1] - 5e-2, tgt[1] + 5e-2, 1e-3),
        tgt[2],
    ]
    kwargs = {"c": c, "complex_image": complex_image}

    # Compute either in chunks or immediately.
    if chunked:
        img = reconstruct.backproject(data.chunk(ping=10), *args, **kwargs)  # type:ignore[arg-type]
        assert isinstance(img.data, dask.array.core.Array)
        assert len(img.data.dask.layers) > 2
        img.load()
    else:
        img = reconstruct.backproject(data, *args, **kwargs)  # type:ignore[arg-type]

    assert isinstance(img.data, np.ndarray)
    assert img.dtype.kind == "f" if real_data and not complex_image else "c"

    # Normalise by the peak.
    img = np.abs(img)  # type: ignore[assignment]
    img /= np.max(img)

    # Check the peak is in the expected position.
    peak = img[np.abs(img).argmax(["x", "y"])]  # type:ignore[call-overload]
    assert np.allclose(peak["x"], 10e-2)
    assert np.allclose(peak["y"], 4e-2)

    # Fit a Gaussian pulse in either direction and check the resolution is not too far
    # from theoretical. As the data uses an impulse, the 'signal' takes the full
    # sampling bandwidth.
    mag_l = np.log(img.loc[{"x": peak["x"] - 1e-3, "y": peak["y"]}])
    mag_r = np.log(img.loc[{"x": peak["x"] + 1e-3, "y": peak["y"]}])
    mag_u = np.log(img.loc[{"x": peak["x"], "y": peak["y"] - 1e-3}])
    mag_d = np.log(img.loc[{"x": peak["x"], "y": peak["y"] + 1e-3}])
    sigma_x = np.sqrt(-1.0 / (mag_l + mag_r)) * 1e-3
    sigma_y = np.sqrt(-1.0 / (mag_d + mag_u)) * 1e-3
    expected = c / (2 * fs)
    assert np.isclose(sigma_x, expected, atol=0, rtol=0.1)
    assert np.isclose(sigma_y, expected, atol=0, rtol=0.1)

    # Check the area away from the peak is low amplitude.
    img.loc[{"x": slice(9e-2, 11e-2), "y": slice(3e-2, 5e-2)}] = 0
    assert np.all(np.abs(img) < 0.1)


@pytest.mark.parametrize("focal_type", ["list", "ndarray", "dataarray"])
def test_backproject_z_array(constant_z_data, focal_type: str):
    data, tx_pos, rx_pos, tgt, c, fs = constant_z_data

    # Focal points: 1D vectors of x and y, and 2D array of z.
    x = np.arange(tgt[0] - 5e-2, tgt[0] + 5e-2, 1e-3)
    y = np.arange(tgt[1] - 5e-2, tgt[1] + 5e-2, 1e-3)
    z = np.full((len(x), len(y)), tgt[2])

    # Convert type as needed.
    x_in: ArrayLike
    y_in: ArrayLike
    z_in: ArrayLike
    if focal_type == "list":
        x_in = x.tolist()
        y_in = y.tolist()
        z_in = z.tolist()
    elif focal_type == "dataarray":
        x_in = xr.DataArray(x, coords=[("focal_x", x)])
        y_in = xr.DataArray(y, coords=[("focal_y", y)])
        z_in = xr.DataArray(z, coords=[x, y])
    else:
        x_in = x
        y_in = y
        z_in = z

    # Compute in chunks.
    data = data.chunk(ping=10)
    img = reconstruct.backproject(data, tx_pos, rx_pos, x_in, y_in, z_in, c=c)
    assert isinstance(img.data, dask.array.core.Array)
    assert len(img.data.dask.layers) > 2
    img.load()
    assert isinstance(img.data, np.ndarray)
    assert img.dtype.kind == "c"

    # Check the dimension names: set by input DataArrays, defaulting to x and y for
    # other types of inputs.
    if focal_type == "dataarray":
        assert set(img.dims) == {"focal_x", "focal_y"}
        img = img.rename(focal_x="x", focal_y="y")
    else:
        assert set(img.dims) == {"x", "y"}

    # Check the dimension values.
    assert np.allclose(img["x"], x)
    assert np.allclose(img["y"], y)

    # Normalise by the peak.
    img = np.abs(img)  # type: ignore[assignment]
    img /= np.max(img)

    # Check the peak is in the expected position.
    peak = img[np.abs(img).argmax(["x", "y"])]  # type:ignore[call-overload]
    assert np.allclose(peak["x"], 10e-2)
    assert np.allclose(peak["y"], 4e-2)

    # Fit a Gaussian pulse in either direction and check the resolution is not too far
    # from theoretical. As the data uses an impulse, the 'signal' takes the full
    # sampling bandwidth.
    mag_l = np.log(img.loc[{"x": peak["x"] - 1e-3, "y": peak["y"]}])
    mag_r = np.log(img.loc[{"x": peak["x"] + 1e-3, "y": peak["y"]}])
    mag_u = np.log(img.loc[{"x": peak["x"], "y": peak["y"] - 1e-3}])
    mag_d = np.log(img.loc[{"x": peak["x"], "y": peak["y"] + 1e-3}])
    sigma_x = np.sqrt(-1.0 / (mag_l + mag_r)) * 1e-3
    sigma_y = np.sqrt(-1.0 / (mag_d + mag_u)) * 1e-3
    expected = c / (2 * fs)
    assert np.isclose(sigma_x, expected, atol=0, rtol=0.1)
    assert np.isclose(sigma_y, expected, atol=0, rtol=0.1)

    # Check the area away from the peak is low amplitude.
    img.loc[{"x": slice(9e-2, 11e-2), "y": slice(3e-2, 5e-2)}] = 0
    assert np.all(np.abs(img) < 0.1)


def test_backproject_error(constant_z_data):
    data, tx, rx, tgt, c, fs = constant_z_data

    with pytest.raises(ValueError, match="x must be one-dimensional"):
        reconstruct.backproject(data, tx, rx, np.zeros((10, 10)), np.zeros(10), 0)
    with pytest.raises(ValueError, match="x must be one-dimensional"):
        reconstruct.backproject(
            data, tx, rx, xr.DataArray(np.zeros((10, 10))), np.zeros(10), 0
        )

    with pytest.raises(ValueError, match="y must be one-dimensional"):
        reconstruct.backproject(data, tx, rx, np.zeros(10), np.zeros((10, 20)), 0)
    with pytest.raises(ValueError, match="y must be one-dimensional"):
        reconstruct.backproject(
            data, tx, rx, np.zeros(10), xr.DataArray(np.zeros((10, 20))), 0
        )

    with pytest.raises(ValueError, match="z must be a float or a 2D array"):
        reconstruct.backproject(data, tx, rx, np.zeros(10), np.zeros(10), [1, 2, 3, 4])
