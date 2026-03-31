from .fx_encoder_pp import load_effects_encoder_plusplus
from .clap import load_clap_audio
from .stito_encoder import load_stito_encoder
from .fx_encoder import load_effects_encoder

def load_audio_encoder(audio_encoder_type):
    if audio_encoder_type == "fxenc_plusplus":
        return load_effects_encoder_plusplus(model_name="default")
    elif audio_encoder_type == "fxenc":
        return load_effects_encoder()
    elif audio_encoder_type == "clap":
        return load_clap_audio()
    elif audio_encoder_type == "stito_encoder":
        return load_stito_encoder(ckpt_path="ckpt/stito_encoder/afx-rep.ckpt")
    else:
        raise ValueError(f"Invalid audio encoder type: {audio_encoder_type}")