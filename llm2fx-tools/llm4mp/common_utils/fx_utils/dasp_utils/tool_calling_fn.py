
def gain(gain_db: float):
    """Apply gain in dB.
    Args:
        gain_db: Gain in dB (-24.0, 24.0)
    """
    pass

def distortion(
    drive_db: float,
    mix: float
    ):
    """soft-clipping distortion with drive control.
    Args:
        drive_db: Drive in dB (0.0, 24.0)
        mix: Wet/dry mix factor (0.0, 1.0)
    """
    pass

def compressor_expander(
    low_cutoff: float,
    high_cutoff: float,
    mix: float,
    low_shelf_comp_thresh: float,
    low_shelf_comp_ratio: float,
    low_shelf_exp_thresh: float,
    low_shelf_exp_ratio: float,
    low_shelf_at: float,
    low_shelf_rt: float,
    mid_band_comp_thresh: float,
    mid_band_comp_ratio: float,
    mid_band_exp_thresh: float,
    mid_band_exp_ratio: float,
    mid_band_at: float,
    mid_band_rt: float,
    high_shelf_comp_thresh: float,
    high_shelf_comp_ratio: float,
    high_shelf_exp_thresh: float,
    high_shelf_exp_ratio: float,
    high_shelf_at: float,
    high_shelf_rt: float,
):
    """
    Multiband (low-shelf, mid-band, high-shelf) Compressor or Expander.
    Args:
        low_cutoff: Crossover frequency (Hz) between low-shelf and mid-band. (20, 300)
        high_cutoff: Crossover frequency (Hz) between mid-band and high-shelf. (2000, 12000)
        mix: Wet/dry mix factor. (0.2, 0.7)
        low_shelf_comp_thresh: Low-shelf compressor threshold in dB. (-60.0, 0.0)
        low_shelf_comp_ratio: Low-shelf compressor ratio. (1.0, 20.0)
        low_shelf_exp_thresh: Low-shelf expander threshold in dB. (-60.0, 0.0)
        low_shelf_exp_ratio: Low-shelf expander ratio. (0.0, 1.0)
        low_shelf_at: Low-shelf attack time in ms. (5.0, 100.0)
        low_shelf_rt: Low-shelf release time in ms. (5.0, 100.0)
        mid_band_comp_thresh: Mid-band compressor threshold in dB. (-60.0, 0.0)
        mid_band_comp_ratio: Mid-band compressor ratio. (1.0, 20.0)
        mid_band_exp_thresh: Mid-band expander threshold in dB. (-60.0, 0.0)
        mid_band_exp_ratio: Mid-band expander ratio. (0.0, 1.0)
        mid_band_at: Mid-band attack time in ms. (5.0, 100.0)
        mid_band_rt: Mid-band release time in ms. (5.0, 100.0)
        high_shelf_comp_thresh: High-shelf compressor threshold in dB. (-60.0, 0.0)
        high_shelf_comp_ratio: High-shelf compressor ratio. (1.0, 20.0)
        high_shelf_exp_thresh: High-shelf expander threshold in dB. (-60.0, 0.0)
        high_shelf_exp_ratio: High-shelf expander ratio. (0.0, 1.0)
        high_shelf_at: High-shelf attack time in ms. (5.0, 100.0)
        high_shelf_rt: High-shelf release time in ms. (5.0, 100.0)
    """
    pass

def limiter(
    threshold: float,
    at: float,
    rt: float,
):
    """Audio limiter to prevent signal peaks from exceeding a threshold.
    Args:
        threshold: Limiter threshold in dB (-60.0, 0.0)
        at: Attack time in ms (5.0, 100.0)
        rt: Release time in ms (5.0, 100.0)
    """
    pass

def panner(
    pan: float,
    mix: float,
):
    """Stereo panner for positioning audio in the stereo field.
    Args:
        pan: Panning position (-1.0, 1.0)
        mix: Wet/dry mix factor (0.0, 1.0)
    """
    pass

def imager(
    width: float,
    mix: float,
):
    """Stereo width control using Mid-Side processing.
    Args:
        width: Stereo width factor. 0.0 = mono, 1.0 = original, >1.0 = wider (0.0, 2.0)
        mix: Wet/dry mix factor (0.0, 1.0)
    """
    pass


def noise_shaped_reverberation(
    band0_gain: float,
    band1_gain: float,
    band2_gain: float,
    band3_gain: float,
    band4_gain: float,
    band5_gain: float,
    band6_gain: float,
    band7_gain: float,
    band8_gain: float,
    band9_gain: float,
    band10_gain: float,
    band11_gain: float,
    band0_decay: float,
    band1_decay: float,
    band2_decay: float,
    band3_decay: float,
    band4_decay: float,
    band5_decay: float,
    band6_decay: float,
    band7_decay: float,
    band8_decay: float,
    band9_decay: float,
    band10_decay: float,
    band11_decay: float,
    mix: float,
):
    """Artificial reverberation using frequency-band noise shaping.
    Args:
        band0_gain: Gain for first octave band (lowpass below 12 Hz) on (0,1).
        band1_gain: Gain for second octave band (31.5 Hz center) on (0,1).
        band2_gain: Gain for third octave band (63 Hz center) on (0,1).
        band3_gain: Gain for fourth octave band (125 Hz center) on (0,1).
        band4_gain: Gain for fifth octave band (250 Hz center) on (0,1).
        band5_gain: Gain for sixth octave band (500 Hz center) on (0,1).
        band6_gain: Gain for seventh octave band (1 kHz center) on (0,1).
        band7_gain: Gain for eighth octave band (2 kHz center) on (0,1).
        band8_gain: Gain for ninth octave band (4 kHz center) on (0,1).
        band9_gain: Gain for tenth octave band (8 kHz center) on (0,1).
        band10_gain: Gain for eleventh octave band (16 kHz center) on (0,1).
        band11_gain: Gain for twelfth octave band (highpass above 18 kHz) on (0,1).
        band0_decay: Decay parameter for first octave band (lowpass below 12 Hz) on (0,1).
        band1_decay: Decay parameter for second octave band (31.5 Hz center) on (0,1).
        band2_decay: Decay parameter for third octave band (63 Hz center) on (0,1).
        band3_decay: Decay parameter for fourth octave band (125 Hz center) on (0,1).
        band4_decay: Decay parameter for fifth octave band (250 Hz center) on (0,1).
        band5_decay: Decay parameter for sixth octave band (500 Hz center) on (0,1).
        band6_decay: Decay parameter for seventh octave band (1 kHz center) on (0,1).
        band7_decay: Decay parameter for eighth octave band (2 kHz center) on (0,1).
        band8_decay: Decay parameter for ninth octave band (4 kHz center) on (0,1).
        band9_decay: Decay parameter for tenth octave band (8 kHz center) on (0,1).
        band10_decay: Decay parameter for eleventh octave band (16 kHz center) on (0,1).
        band11_decay: Decay parameter for twelfth octave band (highpass above 18 kHz) on (0,1).
        mix: Wet/dry mix factor (0.0, 1.0)
    """
    pass


def parametric_eq(
    low_shelf_gain_db: float,
    low_shelf_cutoff_freq: float,
    low_shelf_q_factor: float,
    band0_gain_db: float,
    band0_cutoff_freq: float,
    band0_q_factor: float,
    band1_gain_db: float,
    band1_cutoff_freq: float,
    band1_q_factor: float,
    band2_gain_db: float,
    band2_cutoff_freq: float,
    band2_q_factor: float,
    band3_gain_db: float,
    band3_cutoff_freq: float,
    band3_q_factor: float,
    high_shelf_gain_db: float,
    high_shelf_cutoff_freq: float,
    high_shelf_q_factor: float,
):
    """Six-band Parametric Equalizer. Low-shelf -> Band 1 -> Band 2 -> Band 3 -> Band 4 -> High-shelf.
    Args:
        low_shelf_gain_db: Low-shelf filter gain in dB (-20.0, 20.0).
        low_shelf_cutoff_freq: Low-shelf filter cutoff frequency in Hz (20, 2000).
        low_shelf_q_factor: Low-shelf filter Q-factor (0.1, 6.0).
        band0_gain_db: Band 1 filter gain in dB (-20.0, 20.0).
        band0_cutoff_freq: Band 1 filter cutoff frequency in Hz (80, 2000).
        band0_q_factor: Band 1 filter Q-factor (0.1, 6.0).
        band1_gain_db: Band 2 filter gain in dB (-20.0, 20.0).
        band1_cutoff_freq: Band 2 filter cutoff frequency in Hz (2000, 8000).
        band1_q_factor: Band 2 filter Q-factor (0.1, 6.0).
        band2_gain_db: Band 3 filter gain in dB (-20.0, 20.0).
        band2_cutoff_freq: Band 3 filter cutoff frequency in Hz (8000, 12000).
        band2_q_factor: Band 3 filter Q-factor (0.1, 6.0).
        band3_gain_db: Band 4 filter gain in dB (-20.0, 20.0).
        band3_cutoff_freq: Band 4 filter cutoff frequency in Hz (12000, sample_rate/2 - 1000).
        band3_q_factor: Band 4 filter Q-factor (0.1, 6.0).
        high_shelf_gain_db: High-shelf filter gain in dB (-20.0, 20.0).
        high_shelf_cutoff_freq: High-shelf filter cutoff frequency in Hz (4000, sample_rate/2 - 1000).
        high_shelf_q_factor: High-shelf filter Q-factor (0.1, 6.0).
    """
    pass

FX_TOOL_DICT = {
    "gain": gain,
    "equalizer": parametric_eq,
    "distortion": distortion,
    "compressor_expander": compressor_expander,
    "limiter": limiter,
    "panner": panner,
    "imager": imager,
    "reverb": noise_shaped_reverberation,
}

def main():
    from transformers.utils import get_json_schema
    fx_chain = [gain,distortion, compressor_expander, limiter, panner, imager, noise_shaped_reverberation, parametric_eq]
    for fx in fx_chain:
        fx_schema = get_json_schema(fx)
        print(fx_schema)

if __name__ == "__main__":
    main()
