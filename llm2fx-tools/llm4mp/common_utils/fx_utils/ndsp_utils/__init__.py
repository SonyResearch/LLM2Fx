from .pb_functional import apply_random_pedalboard_distortion, apply_random_pedalboard_delay, apply_random_pedalboard_reverb, apply_random_pedalboard_compressor, apply_random_three_band_eq, apply_random_pedalboard_gain, apply_random_pedalboard_limiter, apply_random_pedalboard_noise_gate, apply_random_stereo_widener, apply_random_panner
from .pb_functional import parametric_eq, pedalboard_noise_gate, pedalboard_compressor, pedalboard_delay, pedalboard_limiter, pedalboard_gain, pedalboard_noise_gate, pedalboard_reverb, pedalboard_distortion, stereo_widener, stereo_panner
from .tool_calling_fn import ndsp_fx_tools
from .constants import pedalboard_fx_list, pedalboard_fx_param_keys, pedalboard_fx_param_ranges

pedalboard_fx_function_dict = {
    "equalizer": parametric_eq,
    "compressor": pedalboard_compressor,
    # "noise_gate": pedalboard_noise_gate,
    "stereo_widener": stereo_widener,
    "panner": stereo_panner,
    "gain": pedalboard_gain,
    "distortion": pedalboard_distortion,
    "reverb": pedalboard_reverb,
    "delay": pedalboard_delay,
    "limiter": pedalboard_limiter,
}

pedalboard_random_fx_module_dict = {
    "distortion": apply_random_pedalboard_distortion,
    "delay": apply_random_pedalboard_delay,
    "reverb": apply_random_pedalboard_reverb,
    "compressor": apply_random_pedalboard_compressor,
    "equalizer": apply_random_three_band_eq,
    "gain": apply_random_pedalboard_gain,
    "limiter": apply_random_pedalboard_limiter,
    # "noise_gate": apply_random_pedalboard_noise_gate,
    "stereo_widener": apply_random_stereo_widener,
    "panner": apply_random_panner,
}