# SPDX-FileCopyrightText: AirSonar contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
import pytest
from scipy.signal import hilbert
import xarray as xr

from airsonar.processing import signals


@pytest.mark.parametrize(
    "f_start,f_stop,duration", [(10e3, 20e3, 10e-3), (20e3, 10e3, 10e-3)]
)
def test_lfm_generation(f_start: float, f_stop: float, duration: float):
    # Sample.
    f_s = 20 * max(f_start, f_stop)
    t = np.arange(0, duration, 1 / f_s)
    s = signals.lfm(t, f_start, f_stop, duration)

    # Calculate instantaneous frequency from analytic signal.
    s_analytic = xr.apply_ufunc(hilbert, s)
    phase = xr.apply_ufunc(np.unwrap, xr.apply_ufunc(np.angle, s_analytic))
    inst_f = phase.differentiate("time", edge_order=2) / (2 * np.pi)

    # Ignore the edge effects at the ends and compare to the theoretical value.
    inst_f = inst_f[100:-100]
    K = (f_stop - f_start) / duration
    expected = f_start + K * inst_f["time"]
    assert np.isclose((inst_f - expected).mean(), 0, atol=1, rtol=0)


@pytest.mark.parametrize("v", [0.0, np.nan, -100])
def test_lfm_invalid_value(v: float):
    t = np.arange(-1e-3, 5e-3, 1e-5)
    s = signals.lfm(t, 10e3, 20e3, 2e-3, invalid=v)
    assert np.allclose(s[t < 0], v, equal_nan=True)
    assert np.allclose(s[t > 2e-3], v, equal_nan=True)


def test_lfm_error():
    with pytest.raises(ValueError, match="multi-dimensional times must be.+DataArray"):
        signals.lfm([[0, 1, 2], [3, 4, 5]], 10e3, 20e3, 1e-3)


@pytest.mark.parametrize(
    "f_start,f_stop,duration", [(10e3, 20e3, 10e-3), (20e3, 10e3, 10e-3)]
)
def test_lpm_generation(f_start: float, f_stop: float, duration: float):
    # Sample.
    f_s = 20 * max(f_start, f_stop)
    t = np.arange(0, duration, 1 / f_s)
    s = signals.lpm(t, f_start, f_stop, duration)

    # Calculate instantaneous frequency from analytic signal.
    s_analytic = xr.apply_ufunc(hilbert, s)
    phase = xr.apply_ufunc(np.unwrap, xr.apply_ufunc(np.angle, s_analytic))
    inst_f = phase.differentiate("time", edge_order=2) / (2 * np.pi)

    # Ignore the edge effects at the ends and compare to the theoretical value.
    inst_f = inst_f[100:-100]
    tau0 = (f_stop / (f_stop - f_start)) * duration
    expected = f_start * tau0 / (tau0 - inst_f["time"])
    assert np.isclose((inst_f - expected).mean(), 0, atol=1, rtol=0)


@pytest.mark.filterwarnings("ignore:invalid value encountered in log")
@pytest.mark.parametrize("v", [0.0, np.nan, -100])
def test_lpm_invalid_value(v: float):
    t = np.arange(-1e-3, 5e-3, 1e-5)
    s = signals.lpm(t, 10e3, 20e3, 2e-3, invalid=v)
    assert np.allclose(s[t < 0], v, equal_nan=True)
    assert np.allclose(s[t > 2e-3], v, equal_nan=True)


def test_lpm_error():
    with pytest.raises(ValueError, match="multi-dimensional times must be.+DataArray"):
        signals.lpm([[0, 1, 2], [3, 4, 5]], 10e3, 20e3, 1e-3)
