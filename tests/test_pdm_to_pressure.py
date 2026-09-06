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


def to_pdm(signal: xr.DataArray) -> xr.DataArray:
    """Generate a PDM waveform of a given signal.

    This uses a fourth-order sigma-delta modulator. If the amplitude goes above ~0.7,
    the output can become unstable.

    Parameters
    ----------
    signal
        The signal to generate the waveform from. This must be one-dimensional.

    Returns
    -------
    pdm
        The PDM waveform generated from the signal. This has the same coordinates as the
        input signal.

    """
    if not signal.ndim == 1:
        raise ValueError("signal must be one-dimensional")

    # Initialise state.
    integrators = np.array([0.0, 0.0, 0.0, 0.0])
    coefficients = np.array([1, 0.5, 0.15, 0.02])
    feedback = 0.0

    # And process each sample.
    pdm_out = np.zeros_like(signal)
    for n in range(len(signal)):
        integrators[0] = signal[n] - feedback
        integrators[1] += integrators[0]
        integrators[2] += integrators[1]
        integrators[3] += integrators[2]

        val = np.sum(integrators * coefficients)
        feedback = 1 if val >= 0 else -1
        pdm_out[n] = 0.5 if val >= 0 else -0.5

    return xr.DataArray(pdm_out, coords=signal.coords)


@pytest.mark.parametrize(
    "f,fs,t_end,stages",
    [
        (10e3, 4.8e6, 1e-3, [8, 6]),
        (8e3, 5e6, 1e-3, [48]),
    ],
)
def test_downsample_integer_1d_sine(f, fs, t_end, stages):
    t = np.arange(0, t_end, 1 / fs)
    pdm = to_pdm(xr.DataArray(0.5 * np.sin(2 * np.pi * f * t), coords=[("time", t)]))

    # Our PDM output is ±0.5, which halves the processed output. Also note we need to
    # increase the allclose tolerance to handle the error from our modulator, and to
    # skip the first few samples which have higher error.
    downsampled = pdm_to_pressure.downsample_integer(pdm, stages)
    expected = 0.25 * np.sin(2 * np.pi * f * downsampled["time"])
    assert downsampled.dtype == pdm.dtype
    xr.testing.assert_allclose(downsampled[20:], expected[20:], rtol=0, atol=5e-3)

    # Check it properly upcasts.
    pdm_int = pdm.astype("f2")
    downsampled = pdm_to_pressure.downsample_integer(pdm_int, stages)
    assert downsampled.dtype == float
    xr.testing.assert_allclose(downsampled[20:], expected[20:], rtol=0, atol=5e-3)

    # Check with different dimension names.
    pdm = pdm.rename(time="sample")
    expected = expected.rename(time="sample")
    downsampled = pdm_to_pressure.downsample_integer(pdm, stages, dim="sample")
    assert downsampled.dtype == pdm.dtype
    xr.testing.assert_allclose(downsampled[20:], expected[20:], rtol=0, atol=5e-3)


@pytest.mark.parametrize("chunked", [False, True])
def test_downsample_integer_2d_sine(chunked):
    # Generate multiple traces.
    t = np.arange(0, 1e-3, 1 / 4.8e6)
    traces = []
    for p in range(7):
        traces.append(
            to_pdm(
                xr.DataArray(
                    0.5 * np.sin(2 * np.pi * p * 1e3 * t), coords=[("time", t)]
                )
            )
        )
    pdm = xr.concat(traces, "ping").assign_coords(ping=np.arange(7))

    # Downsample.
    if chunked:
        downsampled = pdm_to_pressure.downsample_integer(pdm.chunk(ping=3), [8, 6])
        assert downsampled.chunksizes == {"ping": (3, 3, 1), "time": (len(t) // 48,)}
        downsampled.load()
    else:
        downsampled = pdm_to_pressure.downsample_integer(pdm, [8, 6])
    assert not downsampled.chunksizes

    # Our PDM output is ±0.5, which halves the processed output. Also note we need to
    # increase the allclose tolerance to handle the error from our modulator, and to
    # skip the first few samples which have higher error.
    expected_traces: list[xr.DataArray] = []
    for p in range(7):
        expected_traces.append(0.25 * np.sin(2 * np.pi * p * 1e3 * downsampled["time"]))  # type:ignore[arg-type]
    expected = xr.concat(expected_traces, "ping").assign_coords(ping=np.arange(7))
    xr.testing.assert_allclose(
        downsampled[20:], expected[20:], rtol=0, atol=5e-3, check_dim_order=False
    )
