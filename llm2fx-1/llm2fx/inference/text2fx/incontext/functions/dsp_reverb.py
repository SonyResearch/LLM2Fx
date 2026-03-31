import numpy as np
import math
import scipy.signal
from scipy.fft import rfft, irfft
from functools import partial

def noise_shaped_reverberation(
    x,  # Shape: (channels, seq_len)
    sample_rate,
    band0_gain,
    ...
    band11_decay,
    mix
):
    """Artificial reverberation using frequency-band noise shaping for realistic audio effects.
    sample_rate: 44100
    min_band_gain: float = 0.0,
    max_band_gain: float = 1.0,
    min_band_decay: float = 0.0,
    max_band_decay: float = 1.0,
    min_mix: float = 0.0,
    max_mix: float = 1.0,
    This implementation creates realistic spatial ambience by simulating how sound reflects
    and decays differently across frequency bands in real spaces.

    Args:
        x (numpy.ndarray): Input audio signal. Shape (channels, sequence_length).
            Supports mono or stereo audio (1 or 2 channels).

        sample_rate (float): Audio sample rate in Hz.

        band0_gain through band11_gain (float): Gain parameters for each octave band
            from lowest to highest frequency. Values range from 0.0 to 1.0.

        band0_decay through band11_decay (float): Decay parameters for each octave band
            from lowest to highest frequency. Values range from 0.0 to 1.0.

        mix (float): Mix between dry (original) and wet (reverberant) signal.
            Values range from 0.0 to 1.0.
    Returns:
        y (numpy.ndarray): Reverberated audio signal with shape (channels, sequence_length).
    """
    num_samples=65536
    num_bandpass_taps=1023
    assert num_bandpass_taps % 2 == 1, "num_bandpass_taps must be odd"
    chs, seq_len = x.shape
    assert chs <= 2, "only mono/stereo signals are supported"
    # if mono, convert to stereo
    if chs == 1:
        x = np.repeat(x, 2, axis=0)
        chs = 2
    # stack gains and decays into arrays
    band_gains = np.array([
        band0_gain, band1_gain, band2_gain, band3_gain,
        band4_gain, band5_gain, band6_gain, band7_gain,
        band8_gain, band9_gain, band10_gain, band11_gain
    ])
    band_decays = np.array([
        band0_decay, band1_decay, band2_decay, band3_decay,
        band4_decay, band5_decay, band6_decay, band7_decay,
        band8_decay, band9_decay, band10_decay, band11_decay
    ])
    # create the octave band filterbank filters
    filters = octave_band_filterbank(num_bandpass_taps, sample_rate)
    num_bands = filters.shape[0]
    # generate white noise for IR generation
    pad_size = num_bandpass_taps - 1
    wn = np.random.randn(2, num_bands, num_samples + pad_size)
    wn_filt = np.zeros((2, num_bands, num_samples))
    for ch in range(2):
        for band in range(num_bands):
            wn_filt[ch, band] = np.convolve(wn[ch, band], filters[band], mode='valid')
    t = np.linspace(0, 1, num_samples)
    band_decays = (band_decays * 10.0) + 1.0
    for band in range(num_bands):
        env = np.exp(-band_decays[band] * t)
        for ch in range(2):
            wn_filt[ch, band] *= env * band_gains[band]
    w_filt_sum = np.mean(wn_filt, axis=1, keepdims=True)
    y = np.zeros_like(x)
    x_pad = np.pad(x, ((0, 0), (num_samples - 1, 0)))
    for ch in range(chs):
        y[ch] = np.convolve(x_pad[ch], np.flip(w_filt_sum[ch, 0]), mode='valid')
    y = (1 - mix) * x + mix * y
    return y

def octave_band_filterbank(num_taps, sample_rate):
    """Create octave-spaced bandpass filters.
    Args:
        num_taps (int): Number of FIR filter taps
        sample_rate (float): Audio sample rate in Hz
    Returns:
        filters (numpy.ndarray): Filterbank coefficients with shape (num_bands, num_taps)
    """
    bands = [
        31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000,
    ]
    num_bands = len(bands) + 2
    filts = []
    filt = scipy.signal.firwin(
        num_taps,
        12,
        fs=sample_rate,
    )
    filt = np.flip(filt)
    filts.append(filt)
    for fc in bands:
        f_min = fc / np.sqrt(2)
        f_max = fc * np.sqrt(2)
        f_max = np.clip(f_max, a_min=0, a_max=(sample_rate / 2) * 0.999)
        filt = scipy.signal.firwin(
            num_taps,
            [f_min, f_max],
            fs=sample_rate,
            pass_zero=False,
        )
        filt = np.flip(filt)
        filts.append(filt)
    filt = scipy.signal.firwin(num_taps, 18000, fs=sample_rate, pass_zero=False)
    filt = np.flip(filt)
    filts.append(filt)
    filters = np.stack(filts)
    return filters
