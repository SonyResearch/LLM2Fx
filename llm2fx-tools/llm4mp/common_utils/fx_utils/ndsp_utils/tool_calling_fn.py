def distortion(drive_db: int):
    """
    Distortion adds harmonic content and saturation.
    Args:
        drive_db: Distortion amount in dB. range: (0-20), step: 1.0
    """
    pass

def reverb(room_size: float, damping: float, mix_ratio: float, width: float):
    """
    Reverb adds a sense of space and depth to audio.
    Args:
        room_size: Room size. range: (0.1-0.9), step: 0.1
        damping: Reverb decay. range: (0.1-0.9), step: 0.1
        mix_ratio: Dry/wet mix ratio. range: (0.1-0.9), step: 0.1
        width: Stereo width. range: (0.1-0.9), step: 0.1
    """
    pass

def compressor(threshold_db: int, ratio: int, attack_ms: int, release_ms: int):
    """
    Compression reduces the dynamic range by attenuating signals above a threshold.
    Args:
        threshold_db: range: (-40 to -5), step: 1.0
        ratio: range: (1.0-20.0), step: 1.0
        attack_ms: range: (0.0-500), step: 1.0
        release_ms: range: (0.0-1000), step: 25.0
    """
    pass

def delay(delay_seconds: float, feedback: float, mix_ratio: float):
    """
    Delay creates echoes by playing back the audio signal.
    Args:
        delay_seconds: range: (0.00-0.7), step: 0.05
        feedback: range: (0.00-0.6), step: 0.05
        mix_ratio: range: (0.0-1.0), step: 0.1
    """
    pass

def limiter(threshold_db: int, release_ms: int):
    """
    Limiter prevents audio from exceeding a set level.
    Args:
        threshold_db: range: (-20,-1), step: 1.0
        release_ms: range: (0,1000), step: 25.0
    """
    pass

def gain(gain_db: int):
    """
    Gain control adjusts the volume.
    Args:
        gain_db: range: (-20,20), step: 1.0
    """
    pass

def equalizer(
    low_gain_db: int, 
    low_cutoff_freq: int,
    low_q_factor: float,
    mid_gain_db: int,
    mid_cutoff_freq: int,
    mid_q_factor: float,
    high_gain_db: int,
    high_cutoff_freq: int,
    high_q_factor: float,
):
    """
    three-band parametric EQ
    Args:
        low_gain_db: range: (-20,20), step: 1
        low_cutoff_freq: range: (0,400), step: 10
        low_q_factor: range: (0.0,6.0), step: 0.5
        mid_gain_db: range: (-20,20), step: 1.0
        mid_cutoff_freq: range: (250,6000), step: 100
        mid_q_factor: range: (0.0,6.0), step: 0.5
        high_gain_db: range: (-20,20), step: 1.0
        high_cutoff_freq: range: (4000,20000), step: 500
        high_q_factor: range: (0.0,6.0), step: 0.5
    """
    pass

def panner(pan: float):
    """    
    Panning controls the left-right positioning of audio in a stereo mix.
    Args:
        pan: range: (-1.0,1.0), step: 0.1
    """
    pass

def stereo_widener(width: float):
    """
    Stereo widening manipulates the stereo field to make audio sound wider or narrower.
    Args:
        width: range: (0.0,1.5), step: 0.1
    """
    pass

ndsp_fx_tools = {
    "gain": gain,                  # 1 param
    "distortion": distortion,            # 1 param
    "compressor": compressor,            # 4 params
    "delay": delay,                 # 3 params
    "limiter": limiter,               # 2 params
    "panner": panner,                # 1 param
    "stereo_widener": stereo_widener,        # 1 param
    "equalizer": equalizer, # 9 params
    "reverb": reverb, # 3 params
    }
