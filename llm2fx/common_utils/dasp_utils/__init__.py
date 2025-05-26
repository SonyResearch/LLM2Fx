from .functional import (
    gain,
    stereo_bus,
    stereo_panner,
    stereo_widener,
    noise_shaped_reverberation,
    compressor,
    distortion,
    parametric_eq,
)

from .modules import (
    Processor,
    Compressor,
    ParametricEQ,
    NoiseShapedReverb,
    Gain,
    Distortion,
)
# custom modules
from .custom_modules import (
    Distortion,
    Multiband_Compressor,
    Limiter,
)

from .augment import Random_FX_Augmentation