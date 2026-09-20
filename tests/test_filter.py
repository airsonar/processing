# SPDX-FileCopyrightText: AirSonar contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
import pytest
import xarray as xr

from airsonar.processing import filter


def goertzel(
    da: xr.DataArray, target_dim: str = "frequency", sample_dim: str = "time"
) -> tuple[xr.DataArray, xr.DataArray]:
    """Use the Goertzel algorithm to evaluate the DFT at certain frequencies.

    Parameters
    ----------
    da
        The data to apply the algorithm to.
    target_dim
        The dimension containing the frequencies of interest.
    sample_dim
        The dimension containing the sample times.

    Returns
    -------
    magnitude : xr.DataArray
        The magnitude of the DFT for each target frequency.
    phase : xr.DataArray
        The phase of the DFT for each target frequency.

    """
    # Properties of the sampling.
    N = len(da[sample_dim])
    fs = 1 / float(da[sample_dim][1] - da[sample_dim][0])

    # Properties of the target frequencies.
    k = np.round((N * da[target_dim]) / fs)
    omega = 2 * np.pi * k / N
    sine = np.sin(omega)
    cosine = np.cos(omega)

    # Run the IIR filter part of the algorithm.
    coeff = 2 * cosine.values  # type:ignore[attr-defined]
    q0 = q1 = q2 = 0
    data = da.transpose(target_dim, sample_dim).values
    for i in range(N):
        q0 = coeff * q1 - q2 + data[:, i]
        q2 = q1
        q1 = q0

    # And compute the final magnitudes.
    scaling_factor = N / 2
    real = (q1 - q2 * cosine) / scaling_factor
    imag = (q2 * sine) / scaling_factor
    return np.sqrt(real**2 + imag**2), np.arctan2(imag, real)  # type:ignore[return-value]


@pytest.mark.parametrize("chunked", [False, True])
def test_fir_least_squares_lowpass(chunked):
    # Generate an array with different frequency sinusoids.
    _t = np.arange(0, 10e-3, 1 / 100e3)
    _f = np.arange(10, 31) * 1e3
    t = xr.DataArray(_t, coords=[("time", _t)])
    f = xr.DataArray(_f, coords=[("frequency", _f)])
    s: xr.DataArray = np.cos(2 * np.pi * f * t)  # type:ignore[assignment]

    # Filter.
    if chunked:
        filt = filter.fir_least_squares(s.chunk(frequency=10), None, 20e3, 3e3)
        assert filt.chunksizes == {"frequency": (10, 10, 1), "time": (len(_t),)}
        filt.load()
    else:
        filt = filter.fir_least_squares(s, None, 20e3, 3e3)
    assert not filt.chunksizes

    # Check the magnitude and phase of the filter output.
    mag, phase = goertzel(filt)
    _, origphase = goertzel(s)
    pb = slice(None, 20e3)
    sb = slice(23e3, None)
    assert np.allclose(mag.sel(frequency=pb), 1, rtol=0, atol=0.01)
    assert np.allclose(
        phase.sel(frequency=pb), origphase.sel(frequency=pb), rtol=0, atol=0.005
    )
    assert np.allclose(mag.sel(frequency=sb), 0, rtol=0, atol=0.01)


@pytest.mark.parametrize("chunked", [False, True])
def test_fir_least_squares_highpass(chunked):
    # Generate an array with different frequency sinusoids.
    _t = np.arange(0, 10e-3, 1 / 100e3)
    _f = np.arange(10, 31) * 1e3
    t = xr.DataArray(_t, coords=[("time", _t)])
    f = xr.DataArray(_f, coords=[("frequency", _f)])
    s: xr.DataArray = np.cos(2 * np.pi * f * t)  # type:ignore[assignment]

    # Filter.
    if chunked:
        filt = filter.fir_least_squares(s.chunk(frequency=10), 20e3, None, 3e3)
        assert filt.chunksizes == {"frequency": (10, 10, 1), "time": (len(_t),)}
        filt.load()
    else:
        filt = filter.fir_least_squares(s, 20e3, None, 3e3)
    assert not filt.chunksizes

    # Check the magnitude and phase of the filter output.
    mag, phase = goertzel(filt)
    _, origphase = goertzel(s)
    pb = slice(20e3, None)
    sb = slice(None, 17e3)
    assert np.allclose(mag.sel(frequency=pb), 1, rtol=0, atol=0.01)
    assert np.allclose(
        phase.sel(frequency=pb), origphase.sel(frequency=pb), rtol=0, atol=0.005
    )
    assert np.allclose(mag.sel(frequency=sb), 0, rtol=0, atol=0.01)


@pytest.mark.parametrize("chunked", [False, True])
def test_fir_least_squares_bandpass(chunked):
    # Generate an array with different frequency sinusoids.
    _t = np.arange(0, 10e-3, 1 / 100e3)
    _f = np.arange(10, 31.1, 0.5) * 1e3
    t = xr.DataArray(_t, coords=[("time", _t)])
    f = xr.DataArray(_f, coords=[("frequency", _f)])
    s: xr.DataArray = np.cos(2 * np.pi * f * t)  # type:ignore[assignment]

    # Filter.
    if chunked:
        filt = filter.fir_least_squares(s.chunk(frequency=20), 17e3, 22e3, 3e3)
        assert filt.chunksizes == {"frequency": (20, 20, 3), "time": (len(_t),)}
        filt.load()
    else:
        filt = filter.fir_least_squares(s, 17e3, 22e3, 3e3)
    assert not filt.chunksizes

    # Check the magnitude and phase of the filter output.
    mag, phase = goertzel(filt)
    _, origphase = goertzel(s)
    pb = slice(17e3, 22e3)
    sb1 = slice(None, 14e3)
    sb2 = slice(25e3, None)
    assert np.allclose(mag.sel(frequency=pb), 1, rtol=0, atol=0.01)
    assert np.allclose(
        phase.sel(frequency=pb), origphase.sel(frequency=pb), rtol=0, atol=0.005
    )
    assert np.allclose(mag.sel(frequency=sb1), 0, rtol=0, atol=0.01)
    assert np.allclose(mag.sel(frequency=sb2), 0, rtol=0, atol=0.01)


def test_fir_least_squares_errors():
    t = np.arange(0, 1e-3, 1 / 100e3)
    s = xr.DataArray(np.sin(2 * np.pi * 5e3 * t), coords=[("time", t)])

    with pytest.raises(ValueError, match="only one.+low_edge.+high_edge"):
        filter.fir_least_squares(s, None, None, 5e3)

    with pytest.raises(ValueError, match="transition_width.+greater than zero"):
        filter.fir_least_squares(s, 10e3, 20e3, -1e3)
    with pytest.raises(ValueError, match="transition_width.+greater than zero"):
        filter.fir_least_squares(s, 10e3, 20e3, 0)

    with pytest.raises(ValueError, match="high_edge.+higher than the Nyquist"):
        filter.fir_least_squares(s, None, 45e3, 6e3)

    with pytest.raises(ValueError, match="low_edge.+smaller than.+transition"):
        filter.fir_least_squares(s, 5e3, None, 6e3)

    with pytest.raises(ValueError, match="high_edge.+higher than the Nyquist"):
        filter.fir_least_squares(s, 10e3, 45e3, 6e3)
    with pytest.raises(ValueError, match="low_edge.+smaller than.+transition"):
        filter.fir_least_squares(s, 5e3, 40e3, 6e3)


@pytest.mark.parametrize("use_complex", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
def test_matched_filter_lfm(use_complex: bool, chunked: bool):
    # Signal parameters.
    fs = 50e3
    f_start = 10e3
    B = 15e3
    duration = 0.02
    K = B / duration

    # Function to generate traces.
    def s(t):
        phase = 2 * np.pi * f_start * t + np.pi * K * t**2
        if use_complex:
            out = np.exp(1j * phase)
        else:
            out = np.sin(phase)

        out[t < 0] = 0
        out[t > duration] = 0
        return out

    # Generate the signal plus three data traces with different delays.
    t_signal = np.arange(0, 2e-3, 1 / fs)
    signal = xr.DataArray(s(t_signal), coords=[("time", t_signal)])
    t_data = np.arange(0, 10e-3, 1 / fs)
    data = xr.DataArray(
        [s(t_data - 5e-3), s(t_data - 4e-3), s(t_data - 6e-3)],
        coords=[("ping", [1, 2, 3]), ("time", t_data)],
    )

    # Apply the matched filter.
    if chunked:
        matched = filter.matched_filter(data.chunk(ping=2), signal)
        assert matched.chunksizes == {"ping": (2, 1), "time": (len(t_data),)}
        matched.load()
    else:
        matched = filter.matched_filter(data, signal)
    assert not matched.chunksizes

    # Check the peaks of the output occur at the expected times.
    expected_t = xr.DataArray([5e-3, 4e-3, 6e-3], coords=[data["ping"]])
    xr.testing.assert_allclose(
        matched["time"][np.abs(matched).argmax(["time"])].reset_coords(drop=True),  # type:ignore[call-overload]
        expected_t,
    )

    # Check the peak magnitudes correspond to the energy in the signal.
    signal_e = (np.abs(signal) ** 2).sum()
    assert np.allclose(np.abs(matched).max("time"), signal_e)  # type:ignore[call-overload]


def test_matched_filter_error():
    data = xr.DataArray(np.zeros(100), coords=[("time", np.arange(100))])
    signal = xr.DataArray(np.zeros(10), coords=[("time", np.arange(10))])

    with pytest.raises(ValueError, match="unknown method"):
        filter.matched_filter(data, signal, method="super_extra_fast")  # type:ignore[arg-type]
