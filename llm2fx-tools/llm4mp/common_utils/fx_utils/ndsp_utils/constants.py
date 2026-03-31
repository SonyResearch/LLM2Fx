pedalboard_fx_list = [
    "equalizer",
    "compressor",
    "stereo_widener",
    "gain",
    "panner",
    "distortion",
    "reverb",
    "delay",
    "limiter"
]

pedalboard_fx_param_keys = {
    'equalizer': [
        'low_gain_db', 'low_cutoff_freq', 'low_q_factor',
        'mid_gain_db', 'mid_cutoff_freq', 'mid_q_factor',
        'high_gain_db', 'high_cutoff_freq', 'high_q_factor'
    ],
    'compressor': ['threshold_db', 'ratio', 'attack_ms', 'release_ms'],
    'stereo_widener': ['width'],
    'gain': ['gain_db'],
    'panner': ['pan'],
    'distortion': ['drive_db'],
    'reverb': ['room_size', 'damping', 'width','mix_ratio'],
    'delay': ['delay_seconds', 'feedback', 'mix_ratio'],
    'limiter': ['threshold_db', 'release_ms']
}

# Comprehensive parameter ranges and steps for all audio effects
pedalboard_fx_param_ranges = {
    "gain": {
        "gain_db": {
            "min": -20.0,
            "max": 20.0,
            "step": 2.0,
            "scale": "linear",
            "default": 0.0,
            "fine_grained": {"min": -6.0, "max": 6.0, "step": 1.0, "scale": "linear"} # perhaps 0.5 if we want to be more precised. 1 dB could be better for stems within a mixture
        }
    },
    "stereo_widener": {
        "width": {
            "min": 0.0,
            "max": 1.5,
            "step": 0.1,
            "scale": "linear",
            "default": 0.0,
            "fine_grained": {"min": 1.1, "max": 1.5, "step": 0.1, "scale": "linear"} # this one is difficult as it depends on the type of stereo widener, but 0.1 seems ok (given that step is also 0.1 we don't do fine grain here right ?)
        }
    },
    "panner": {
        "pan": {
            "min": -1.0,
            "max": 1.0,
            "step": 0.1,
            "scale": "linear",
            "default": 0.0,
            "fine_grained": {"min": -0.6, "max": 0.6, "step": 0.1, "scale": "linear"} # okay !
        }
    },
    "equalizer": {
        "low_gain_db": {
            "min": -20.0,
            "max": 20.0,
            "step": 2.0,
            "scale": "linear",
            "default": 0.0,
            "fine_grained": {"min": -6.0, "max": 6.0, "step": 1.0, "scale": "linear"} # i think that for all eq gain we should do the same as for gain, and
        },
        "low_cutoff_freq": {
            "min": 0.0,
            "max": 400.0,
            "step": 20.0,
            "scale": "linear",
            "default": 200.0,
            "fine_grained": {"min": 60.0, "max": 120.0, "step": 10.0, "scale": "linear"} # okay !
        },
        "low_q_factor": {
            "min": 0.0,
            "max": 6.0,
            "step": 0.5,
            "default": 0.0,
            "scale": "linear",
            "fine_grained": {"min": 0.5, "max": 3, "step": 0.25, "scale": "linear"} # i feel people dont do big steps when changing Q. maybe 0.25 ? also lets have the same for all Qs
        },
        "mid_gain_db": {
            "min": -20.0,
            "max": 20.0,
            "step": 2.0,
            "scale": "linear",
            "default": 0.0,
            "fine_grained": {"min": -6.0, "max": 6.0, "step": 1.0, "scale": "linear"}
        },
        "mid_cutoff_freq": {
            "min": 250.0,
            "max": 6000.0,
            "step": 250.0,
            "scale": "linear",
            "default": 2000.0,
            "fine_grained": {"min": 250.0, "max": 1000.0, "step": 100.0, "scale": "linear"} # step=100.0 
        },
        "mid_q_factor": {
            "min": 0.1,
            "max": 6.0,
            "step": 0.5,
            "scale": "linear",
            "default": 0.0,
            "fine_grained": {"min": 0.5, "max": 3, "step": 0.25, "scale": "linear"}
        },
        "high_gain_db": {
            "min": -20.0,
            "max": 20.0,
            "step": 2.0,
            "scale": "linear",
            "default": 0.0,
            "fine_grained": {"min": -6.0, "max": 6.0, "step": 1.0, "scale": "linear"}
        },
        "high_cutoff_freq": {
            "min": 4000.0,
            "max": 20000.0,
            "step": 1000.0,
            "scale": "linear",
            "default": 10000.0,
            "fine_grained": {"min": 4000.0, "max": 8000.0, "step": 500.0, "scale": "linear"} # min is 4k ?, the rest i think is okay
        },
        "high_q_factor": {
            "min": 0.0,
            "max": 6.0,
            "step": 0.5,
            "scale": "linear",
            "default": 0.0,
            "fine_grained": {"min": 0.5, "max": 3, "step": 0.5, "scale": "linear"}
        }
    },
    "distortion": {
        "drive_db": {
            "min": 0.0,
            "max": 20.0,
            "step": 2.0,
            "scale": "linear",
            "default": 0.0,
            "fine_grained": {"min": 1.0, "max": 5.0, "step": 0.5, "scale": "linear"} # it depends on the type of distortion, but i would say people is more precise when dealing with less more subtle distortion, thus; {"min": 1.0, "max": 5.0, "step": 0.5}
        }
    },
    "delay": {
        "delay_seconds": {
            "min": 0.0,
            "max": 0.7,
            "step": 0.05,
            "scale": "linear",
            "default": 0.0,
            "fine_grained": {"min": 0.01, "max": 0.2, "step": 0.02, "scale": "linear"} #tbh i am a bit uncertain since this depends a lot on the creative purposes of the delay and people often synchronize the delay time with the bpm, i would say 0.02 is better, 0.01 is too small
        },
        "feedback": {
            "min": 0.0,
            "max": 0.6,
            "step": 0.05,
            "scale": "linear",
            "default": 0.0,
            "fine_grained": {"min": 0.01, "max": 0.2, "step": 0.02, "scale": "linear"}
        },
        "mix_ratio": {
            "min": 0.0,
            "max": 1.0,
            "step": 0.1,
            "scale": "linear",
            "default": 0.0,
            "fine_grained": {"min": 0.1, "max": 1.0, "step": 0.1, "scale": "linear"}
        }
    },
    "reverb": {
        "room_size": {
            "min": 0.0,
            "max": 0.9,
            "step": 0.1, # 0.1
            "scale": "linear",
            "default": 0.0,
            "fine_grained": {"min": 0.3, "max": 0.6, "step": 0.05, "scale": "linear"}
        },
        "damping": {
            "min": 0.0,
            "max": 0.9,
            "step": 0.1, # 0.1
            "scale": "linear",
            "default": 0.0,
            "fine_grained": {"min": 0.3, "max": 0.6, "step": 0.05, "scale": "linear"}
        },
        "width": {
            "min": 0.0,
            "max": 0.9,
            "step": 0.1, 
            "scale": "linear",
            "default": 0.0,
            "fine_grained": {"min": 0.3, "max": 0.6, "step": 0.05, "scale": "linear"}
        },
        "mix_ratio": {
            "min": 0.0,
            "max": 1.0,
            "step": 0.1,
            "scale": "linear",
            "default": 0.0,
            "fine_grained": {"min": 0.1, "max": 1.0, "step": 0.1, "scale": "linear"}
        }
    },
    "compressor": {
        "threshold_db": {
            "min": -40.0,
            "max": -5.0,
            "step": 5.0,
            "scale": "linear",
            "default": 0.0,
            "fine_grained": {"min": -20.0, "max": -10.0, "step": 1.0, "scale": "linear"} # okay !
        },
        "ratio": {
            "min": 1.0,
            "max": 20.0, # i would increase this to 20
            "step": 1.0, # increase to 1
            "default": 0.0,
            "scale": "linear",
            "fine_grained": {"min": 2.0, "max": 8.0, "step": 0.5, "scale": "linear"} # {"min": 2.0, "max": 8.0, "step": 0.5}
        },
        "attack_ms": {
            "min": 0.0,
            "max": 500.0,
            "step": 5.0, # i feel this shouldnt be linear, maybe log increase ? 
            "default": 250.0,
            "scale": "log",
            "fine_grained": {"min": 1.0, "max": 30.0, "step": 1.0, "scale": "linear"} # okay !
        },
        "release_ms": {
            "min": 0.0,
            "max": 1000.0,
            "step": 50.0, # also lets do it log increase
            "default": 500.0,
            "scale": "log",
            "fine_grained": {"min": 0.0, "max": 500.0, "step": 25.0, "scale": "linear"} #  {"min": 50.0, "max": 500.0, "step": 25.0}
        }
    },
    "limiter": {
        "threshold_db": {
            "min": -20.0,
            "max": -1.0,
            "step": 1.0,
            "scale": "linear",
            "default": -1.0,
            "fine_grained": {"min": -5.0, "max": -1.0, "step": 0.1, "scale": "linear"} # {"min": -5.0, "max": -1.0, "step": 0.1}
        },
        "release_ms": {
            "min": 0.0,
            "max": 1000.0,
            "step": 50.0, # log too !
            "default": 500.0,
            "scale": "log",
            "fine_grained": {"min": 0.0, "max": 300.0, "step": 25.0, "scale": "linear"}
        }
    }
}