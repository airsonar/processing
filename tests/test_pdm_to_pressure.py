# SPDX-FileCopyrightText: AirSonar contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
import pytest
import xarray as xr

from airsonar.processing import pdm_to_pressure


@pytest.mark.parametrize("high,low", [(1, 0), (1, -0.5), (1, -1), (8, 4), (-4, -8)])
def test_binary_to_float_nrz_1d(high, low):
    binary = xr.DataArray(
        [high, low, low, low, high, high, low, high], coords=[("time", np.arange(8))]
    )
    nrz = pdm_to_pressure.binary_to_float_nrz(binary)
    assert nrz.dims == ("time",)
    assert np.all(nrz["time"] == binary["time"])
    assert np.allclose(nrz, [0.5, -0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5])

    nrz = pdm_to_pressure.binary_to_float_nrz(binary, invert=True)
    assert nrz.dims == ("time",)
    assert np.all(nrz["time"] == binary["time"])
    assert np.allclose(nrz, [-0.5, 0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5])

    nrz = pdm_to_pressure.binary_to_float_nrz(binary, high_value=1.3)
    assert nrz.dims == ("time",)
    assert np.all(nrz["time"] == binary["time"])
    assert np.allclose(nrz, [1.3, -1.3, -1.3, -1.3, 1.3, 1.3, -1.3, 1.3])

    nrz = pdm_to_pressure.binary_to_float_nrz(binary, high_value=4, invert=True)
    assert nrz.dims == ("time",)
    assert np.all(nrz["time"] == binary["time"])
    assert np.allclose(nrz, [-4, 4, 4, 4, -4, -4, 4, -4])


def test_binary_to_float_nrz_2d():
    binary = xr.DataArray(
        [
            [1, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 1],
            [1, 0, 1, 0, 1, 0],
        ],
        coords=[("ping", [1, 2, 3]), ("time", np.arange(6))],
    )
    expected = xr.DataArray(
        [
            [0.5, 0.5, -0.5, -0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5, 0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5, 0.5, -0.5],
        ],
        coords=[("ping", [1, 2, 3]), ("time", np.arange(6))],
    )

    nrz = pdm_to_pressure.binary_to_float_nrz(binary)
    xr.testing.assert_allclose(nrz, expected, check_dim_order=False)


def test_binary_to_float_nrz_error():
    data = xr.DataArray([1.2, 0.5, 0.4, 0.3, 1.8], coords=[("time", np.arange(5))])
    with pytest.raises(ValueError, match="only possible with binary"):
        pdm_to_pressure.binary_to_float_nrz(data)
