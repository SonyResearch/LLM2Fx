from .dasp_modules import Gain, ParametricEQ, NoiseShapedReverb, Compressor
from .custom_modules import Distortion, Limiter, Imager, Panner

dasp_fx_list = [
    'equalizer','distortion','compressor',
    'gain', 'panner',
    'imager', 'reverb','limiter'
]

dasp_fx_module_dict = {
    "equalizer": ParametricEQ(sample_rate=44100),
    "distortion": Distortion(sample_rate=44100),
    "compressor": Compressor(sample_rate=44100),
    "limiter": Limiter(sample_rate=44100),
    "imager": Imager(sample_rate=44100),
    "gain": Gain(sample_rate=44100),
    "panner": Panner(sample_rate=44100),
    "reverb": NoiseShapedReverb(sample_rate=44100),
}

dasp_fx_param_keys = {
    "equalizer": ['low_shelf_gain_db','low_shelf_cutoff_freq','low_shelf_q_factor','band0_gain_db','band0_cutoff_freq','band0_q_factor','band1_gain_db','band1_cutoff_freq','band1_q_factor','band2_gain_db','band2_cutoff_freq','band2_q_factor','band3_gain_db','band3_cutoff_freq','band3_q_factor','high_shelf_gain_db','high_shelf_cutoff_freq','high_shelf_q_factor'],
    "distortion": ['drive_db', 'mix'],
    "compressor": ['threshold_db', 'ratio', 'attack_ms', 'release_ms', 'knee_db', 'makeup_gain_db'],
    "gain": ['gain_db'],
    "panner": ['pan', 'mix'],
    "imager": ['width', 'mix'],
    "reverb": ['band0_gain', 'band1_gain', 'band2_gain', 'band3_gain', 'band4_gain', 'band5_gain', 'band6_gain', 'band7_gain', 'band8_gain', 'band9_gain', 'band10_gain', 'band11_gain', 'band0_decay', 'band1_decay', 'band2_decay', 'band3_decay', 'band4_decay', 'band5_decay', 'band6_decay', 'band7_decay', 'band8_decay', 'band9_decay', 'band10_decay', 'band11_decay', 'mix'],
    "limiter": ['threshold', 'at', 'rt']
}