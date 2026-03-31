import numpy as np
import math
import scipy.signal
from scipy.fft import rfft, irfft
from functools import partial

def parametric_eq(
    x,  # Shape: (channels, seq_len)
    sample_rate,
    low_shelf_gain_db,
    low_shelf_cutoff_freq,
    low_shelf_q_factor,
    band0_gain_db,
    band0_cutoff_freq,
    band0_q_factor,
    band1_gain_db,
    band1_cutoff_freq,
    band1_q_factor,
    band2_gain_db,
    band2_cutoff_freq,
    band2_q_factor,
    band3_gain_db,
    band3_cutoff_freq,
    band3_q_factor,
    high_shelf_gain_db,
    high_shelf_cutoff_freq,
    high_shelf_q_factor,
):
    """Six-band Parametric Equalizer for audio signal processing.
    sample_rate: 44100
    min_gain_db: -20.0
    max_gain_db: 20.0
    min_q_factor: 0.1
    max_q_factor: 6.0
    param_ranges = {
            "low_shelf_gain_db": (min_gain_db, max_gain_db),
            "low_shelf_cutoff_freq": (20, 2000),
            "low_shelf_q_factor": (min_q_factor, max_q_factor),
            "band0_gain_db": (min_gain_db, max_gain_db),
            "band0_cutoff_freq": (80, 2000),
            "band0_q_factor": (min_q_factor, max_q_factor),
            "band1_gain_db": (min_gain_db, max_gain_db),
            "band1_cutoff_freq": (2000, 8000),
            "band1_q_factor": (min_q_factor, max_q_factor),
            "band2_gain_db": (min_gain_db, max_gain_db),
            "band2_cutoff_freq": (8000, 12000),
            "band2_q_factor": (min_q_factor, max_q_factor),
            "band3_gain_db": (min_gain_db, max_gain_db),
            "band3_cutoff_freq": (12000, (sample_rate // 2) - 1000),
            "band3_q_factor": (min_q_factor, max_q_factor),
            "high_shelf_gain_db": (min_gain_db, max_gain_db),
            "high_shelf_cutoff_freq": (4000, (sample_rate // 2) - 1000),
            "high_shelf_q_factor": (min_q_factor, max_q_factor),
    }
    This function implements a full parametric EQ with six bands in the following configuration:
    Low-shelf -> Band 1 -> Band 2 -> Band 3 -> Band 4 -> High-shelf

    Each band is implemented as a biquad filter with adjustable gain, frequency, and Q-factor,
    allowing precise control over the audio's frequency response. The EQ is implemented
    as a cascade of second-order sections for optimal numerical stability.

    Args:
        x (numpy.ndarray): Time domain audio array with shape (channels, sequence_length)
        sample_rate (float): Audio sample rate in Hz.

        low_shelf_gain_db (float): Low-shelf filter gain in dB. Controls bass boost/cut.
        low_shelf_cutoff_freq (float): Low-shelf filter cutoff frequency in Hz.
        low_shelf_q_factor (float): Low-shelf filter Q-factor. Controls transition steepness.

        band0_gain_db to band3_gain_db (float): Band filter gains in dB.
        band0_cutoff_freq to band3_cutoff_freq (float): Band filter center frequencies in Hz.
        band0_q_factor to band3_q_factor (float): Band filter Q-factors. Controls bandwidth.

        high_shelf_gain_db (float): High-shelf filter gain in dB. Controls treble boost/cut.
        high_shelf_cutoff_freq (float): High-shelf filter cutoff frequency in Hz.
        high_shelf_q_factor (float): High-shelf filter Q-factor. Controls transition steepness.

    Returns:
        y (numpy.ndarray): Filtered audio signal with the same shape as the input.
    """
    chs, seq_len = x.shape
    sos = np.zeros((6, 6))
    b, a = biquad(
        low_shelf_gain_db,
        low_shelf_cutoff_freq,
        low_shelf_q_factor,
        sample_rate,
        "low_shelf",
    )
    sos[0, :] = np.concatenate((b, a))
    b, a = biquad(
        band0_gain_db,
        band0_cutoff_freq,
        band0_q_factor,
        sample_rate,
        "peaking",
    )
    sos[1, :] = np.concatenate((b, a))
    b, a = biquad(
        band1_gain_db,
        band1_cutoff_freq,
        band1_q_factor,
        sample_rate,
        "peaking",
    )
    sos[2, :] = np.concatenate((b, a))
    b, a = biquad(
        band2_gain_db,
        band2_cutoff_freq,
        band2_q_factor,
        sample_rate,
        "peaking",
    )
    sos[3, :] = np.concatenate((b, a))
    b, a = biquad(
        band3_gain_db,
        band3_cutoff_freq,
        band3_q_factor,
        sample_rate,
        "peaking",
    )
    sos[4, :] = np.concatenate((b, a))
    b, a = biquad(
        high_shelf_gain_db,
        high_shelf_cutoff_freq,
        high_shelf_q_factor,
        sample_rate,
        "high_shelf",
    )
    sos[5, :] = np.concatenate((b, a))

    x_out = sosfilt_via_fsm(sos, x)

    return x_out

def biquad(
    gain_db: float,
    cutoff_freq: float,
    q_factor: float,
    sample_rate: float,
    filter_type: str = "peaking",
):
    """Implements a digital biquad filter with various filter types for audio processing.

    A biquad filter is a second-order recursive linear filter with two poles and two zeros.
    This implementation includes several common audio filter types used in digital signal processing.

    Args:
        gain_db (float): Gain in decibels.
            For peaking and shelving filters, this controls the boost/cut amount.
            For lowpass/highpass filters, this parameter is used but typically set to 0dB.

        cutoff_freq (float): Filter cutoff/center frequency in Hz.
            This is the frequency at which the filter takes effect.

        q_factor (float): Quality factor.
            Controls filter bandwidth: higher values = narrower filter

        sample_rate (float): Audio sample rate in Hz.
            Used to normalize frequency for digital filter computation.

        filter_type (str, optional): Type of biquad filter to implement. Defaults to "peaking".
            Supported types:
            - "peaking": Bell/parametric EQ filter that boosts/cuts around center frequency
            - "low_shelf": Boosts/cuts frequencies below cutoff, flat above
            - "high_shelf": Boosts/cuts frequencies above cutoff, flat below
            - "low_pass": Passes signals below cutoff, attenuates above
            - "high_pass": Passes signals above cutoff, attenuates below

    Returns:
        tuple: A tuple containing:
            - b (numpy.ndarray): Numerator coefficients [b0, b1, b2].
            - a (numpy.ndarray): Denominator coefficients [a0, a1, a2] where a0 = 1.
    """
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * (cutoff_freq / sample_rate)
    alpha = math.sin(w0) / (2 * q_factor)
    cos_w0 = math.cos(w0)
    sqrt_A = math.sqrt(A)
    if filter_type == "high_shelf":
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "low_shelf":
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + (alpha / A)
        a1 = -2 * cos_w0
        a2 = 1 - (alpha / A)
    elif filter_type == "low_pass":
        b0 = (1 - cos_w0) / 2
        b1 = 1 - cos_w0
        b2 = (1 - cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    elif filter_type == "high_pass":
        b0 = (1 + cos_w0) / 2
        b1 = -(1 + cos_w0)
        b2 = (1 + cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    return b, a

def fft_freqz(b, a, n_fft=512):
    """Compute frequency response using FFT method."""
    B = rfft(b, n_fft)
    A = rfft(a, n_fft)
    H = B / A
    return H

def fft_sosfreqz(sos, n_fft=512):
    """Compute the complex frequency response via FFT of cascade of biquads.
    Args:
        sos (numpy.ndarray): Second order filter sections with shape (n_sections, 6)
        n_fft (int): FFT size. Default: 512

    Returns:
        H (numpy.ndarray): Overall complex frequency response
    """
    n_sections = sos.shape[0]
    H = None

    for section_idx in range(n_sections):
        b = sos[section_idx, :3]
        a = sos[section_idx, 3:]
        if section_idx == 0:
            H = fft_freqz(b, a, n_fft=n_fft)
        else:
            H *= fft_freqz(b, a, n_fft=n_fft)
    return H

def freqdomain_fir(x, H, n_fft):
    """Filter signal in frequency domain."""
    X = rfft(x, n_fft)
    Y = X * H
    y = irfft(Y, n_fft)
    return y

def sosfilt_via_fsm(sos, x):
    """Use the frequency sampling method to approximate a cascade of second order IIR filters.

    Args:
        sos (numpy.ndarray): Tensor of coefficients with shape (n_sections, 6).
        x (numpy.ndarray): Time domain signal with shape (channels, timesteps)

    Returns:
        y (numpy.ndarray): Filtered time domain signal with shape (channels, timesteps)
    """
    n_fft = 2 ** math.ceil(math.log2(x.shape[-1] + x.shape[-1] - 1))
    n_fft = int(n_fft)
    H = fft_sosfreqz(sos, n_fft=n_fft)
    if x.ndim > 1:
        H = H[np.newaxis, :]
    y = freqdomain_fir(x, H, n_fft)
    y = y[..., :x.shape[-1]]
    return y
