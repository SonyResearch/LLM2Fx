import os
import yaml
import torch
import sys
import argparse
import time
import numpy as np
import librosa
import julius
from glob import glob
from tqdm import tqdm
from collections import OrderedDict
from networks import Effects_Encoder

def load_effects_encoder(ckpt_path, device='cuda'):
    """Load and initialize the effects encoder model.

    Args:
        ckpt_path (str): Path to the model checkpoint
        device (str): Device to load the model on

    Returns:
        Effects_Encoder: Initialized and loaded model
    """
    with open(os.path.join(os.path.dirname(ckpt_path), 'configs.yaml'), 'r') as f:
        configs = yaml.full_load(f)
    cfg_enc = configs['Effects_Encoder']['default']

    effects_encoder = Effects_Encoder(cfg_enc)
    reload_weights(effects_encoder, ckpt_path, device)
    effects_encoder.to(device)
    effects_encoder.eval()

    return effects_encoder

def reload_weights(model, ckpt_path, device):
    """Reload model weights from checkpoint.

    Args:
        model: Model to load weights into
        ckpt_path (str): Path to checkpoint file
        device (str): Device to load weights on
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in checkpoint["model"].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)

def convert_audio(wav: torch.Tensor, from_rate: float, to_rate: float) -> torch.Tensor:
    """Convert audio to new sample rate.

    Args:
        wav (torch.Tensor): Input audio tensor
        from_rate (float): Original sample rate
        to_rate (float): Target sample rate

    Returns:
        torch.Tensor: Resampled audio tensor
    """
    if from_rate != to_rate:
        wav = julius.resample_frac(wav, int(from_rate), int(to_rate))
    return wav

def extract_fxenc_features(target_dir, output_dir, ckpt_dir, ckpt_name, device="cuda", audio_extension='wav', sample_rate=44100):
    """Extract features from audio files using the effects encoder.

    Args:
        target_dir (str): Directory containing audio files
        output_dir (str): Directory to save extracted features
        ckpt_dir (str): Directory containing model checkpoints
        ckpt_name (str): Name of checkpoint to use
        device (str): Device to run inference on
        audio_extension (str): Audio file extension
        sample_rate (int): Target sample rate
    """
    model = load_effects_encoder(os.path.join(ckpt_dir, f"fxenc_{ckpt_name}.pt"), device=device)
    target_file_paths = glob(os.path.join(target_dir, '**', f'*.{audio_extension}'), recursive=True)

    start_time = time.time()
    for target_file_path in tqdm(target_file_paths):
        # Load and resample audio
        cur_audio, input_sr = librosa.load(target_file_path, mono=False, sr=None)
        cur_audio = convert_audio(wav=torch.from_numpy(cur_audio),
                                from_rate=input_sr,
                                to_rate=sample_rate).unsqueeze(0).to(device)

        # Extract features
        with torch.no_grad():
            embedding = model(cur_audio)
        audio_embed = embedding.squeeze().detach().cpu().numpy()

        # Save features
        cur_output_path = target_file_path.replace(target_dir, output_dir).replace(
            f'.{audio_extension}', f'_fxenc_{ckpt_name}_embedding.npy')
        os.makedirs(os.path.dirname(cur_output_path), exist_ok=True)
        np.save(cur_output_path, audio_embed)

    print(f"Extraction finished. Time taken: {time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str)
    parser.add_argument('--output_dir', type=str, help='if no output_dir is specified (None), the results will be saved inside the target_dir')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='path to checkpoint weights')
    parser.add_argument('--using_gpu', type=str, default='0')
    parser.add_argument('--device', type=str, default='cuda', help="if this option is not set to 'cpu', inference will happen on gpu only if there is a detected one")
    parser.add_argument('--audio_extension', type=str, default='wav')
    parser.add_argument('--fxenc_ckpt_name', type=str, default='default', choices=['default'])
    args = parser.parse_args()
    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.dirname(__file__)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.using_gpu
    extract_fxenc_features(
        target_dir=args.target_dir,
        output_dir=args.output_dir,
        ckpt_dir=args.ckpt_dir,
        ckpt_name=args.fxenc_ckpt_name,
        device=args.device,
        audio_extension=args.audio_extension
    )
