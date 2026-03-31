"""
    Implementation of differentiable mastering effects based on DASP-pytorch and torchcomp libraries
        - Distortion
        - Multiband Compressor
        - Limiter
    DASP-pytorch: https://github.com/csteinmetz1/dasp-pytorch
    torchcomp: https://github.com/yoyololicon/torchcomp
    DiffVox: https://github.com/SonyResearch/diffvox
"""
from .dasp_signal import biquad, sosfilt_via_fsm
from .dasp_modules import Processor
import torchcomp
from torchcomp import ms2coef, coef2ms, db2amp
from torch_fftconv import fft_conv1d
from torchaudio.transforms import Spectrogram, InverseSpectrogram
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import time
import math
EPS = 1e-7

class Distortion(Processor):
    def __init__(
        self,
        sample_rate: int,
        min_gain_db: float = 0.0,
        max_gain_db: float = 24.0,
        min_mix: float = 0.2,
        max_mix: float = 0.7,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.process_fn = distortion
        self.param_ranges = {
            "drive_db": (min_gain_db, max_gain_db),
            "mix": (min_mix, max_mix),
        }
        self.num_params = len(self.param_ranges)

def distortion(x: torch.Tensor,
    sample_rate: int,
    drive_db: torch.Tensor,
    mix: torch.Tensor):
    """Simple soft-clipping distortion with drive control.
    Args:
        x (torch.Tensor): Input audio tensor with shape (bs, chs, seq_len)
        sample_rate (int): Audio sample rate.
        drive_db (torch.Tensor): Drive in dB with shape (bs)
    Returns:
        torch.Tensor: Output audio tensor with shape (bs, chs, seq_len)
    """
    bs, chs, seq_len = x.size()
    mix = mix.view(-1, 1, 1)
    # return torch.tanh(x * (10 ** (drive_db.view(bs, chs, -1) / 20.0))) -> wrong?
    x_dist = torch.tanh(x * (10 ** (drive_db.view(bs, 1, 1) / 20.0)))
    # parallel compuatation
    return mix * x_dist + (1-mix) * x


class CompressorExpander(Processor):
    def __init__(
        self,
        sample_rate: int,
        min_threshold_db_comp: float = -60.0,
        max_threshold_db_comp: float = 0.0-EPS,
        min_ratio_comp: float = 1.0+EPS,
        max_ratio_comp: float = 20.0,
        min_attack_ms_comp: float = 5.0,
        max_attack_ms_comp: float = 100.0,
        min_release_ms_comp: float = 5.0,
        max_release_ms_comp: float = 100.0,
        min_threshold_db_exp: float = -60.0,
        max_threshold_db_exp: float = 0.0-EPS,
        min_ratio_exp: float = 0.0+EPS,
        max_ratio_exp: float = 1.0-EPS,
        min_attack_ms_exp: float = 5.0,
        max_attack_ms_exp: float = 100.0,
        min_release_ms_exp: float = 5.0,
        max_release_ms_exp: float = 100.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.process_fn = compressor_expander
        self.param_ranges = {
            "low_cutoff": (20, 300),
            "high_cutoff": (2000, 12000),
            "mix": (0.2, 0.7),
            "low_shelf_comp_thresh": (min_threshold_db_comp, max_threshold_db_comp),
            "low_shelf_comp_ratio": (min_ratio_comp, max_ratio_comp),
            "low_shelf_exp_thresh": (min_threshold_db_exp, max_threshold_db_exp),
            "low_shelf_exp_ratio": (min_ratio_exp, max_ratio_exp),
            "low_shelf_at": (min_attack_ms_exp, max_attack_ms_exp),
            "low_shelf_rt": (min_release_ms_exp, max_release_ms_exp),
            "mid_band_comp_thresh": (min_threshold_db_comp, max_threshold_db_comp),
            "mid_band_comp_ratio": (min_ratio_comp, max_ratio_comp),
            "mid_band_exp_thresh": (min_threshold_db_exp, max_threshold_db_exp),
            "mid_band_exp_ratio": (min_ratio_exp, max_ratio_exp),
            "mid_band_at": (min_attack_ms_exp, max_attack_ms_exp),
            "mid_band_rt": (min_release_ms_exp, max_release_ms_exp),
            "high_shelf_comp_thresh": (min_threshold_db_comp, max_threshold_db_comp),
            "high_shelf_comp_ratio": (min_ratio_comp, max_ratio_comp),
            "high_shelf_exp_thresh": (min_threshold_db_exp, max_threshold_db_exp),
            "high_shelf_exp_ratio": (min_ratio_exp, max_ratio_exp),
            "high_shelf_at": (min_attack_ms_exp, max_attack_ms_exp),
            "high_shelf_rt": (min_release_ms_exp, max_release_ms_exp),
        }
        self.num_params = len(self.param_ranges)

def compressor_expander(
    x: torch.Tensor,
    sample_rate: float,

    low_cutoff: torch.Tensor,
    high_cutoff: torch.Tensor,
    mix: torch.Tensor,

    low_shelf_comp_thresh: torch.Tensor,
    low_shelf_comp_ratio: torch.Tensor,
    low_shelf_exp_thresh: torch.Tensor,
    low_shelf_exp_ratio: torch.Tensor,
    low_shelf_at: torch.Tensor,
    low_shelf_rt: torch.Tensor,

    mid_band_comp_thresh: torch.Tensor,
    mid_band_comp_ratio: torch.Tensor,
    mid_band_exp_thresh: torch.Tensor,
    mid_band_exp_ratio: torch.Tensor,
    mid_band_at: torch.Tensor,
    mid_band_rt: torch.Tensor,

    high_shelf_comp_thresh: torch.Tensor,
    high_shelf_comp_ratio: torch.Tensor,
    high_shelf_exp_thresh: torch.Tensor,
    high_shelf_exp_ratio: torch.Tensor,
    high_shelf_at: torch.Tensor,
    high_shelf_rt: torch.Tensor,
):
    """Multiband (Three-band) Compressor.

    Low-shelf -> Mid-band -> High-shelf

    Args:
        x (torch.Tensor): Time domain tensor with shape (bs, chs, seq_len)
        sample_rate (float): Audio sample rate.
        low_cutoff (torch.Tensor): Low-shelf filter cutoff frequency in Hz.
        high_cutoff (torch.Tensor): High-shelf filter cutoff frequency in Hz.
        low_shelf_comp_thresh (torch.Tensor):
        low_shelf_comp_ratio (torch.Tensor):
        low_shelf_exp_thresh (torch.Tensor):
        low_shelf_exp_ratio (torch.Tensor):
        low_shelf_at (torch.Tensor):
        low_shelf_rt (torch.Tensor):
        mid_band_comp_thresh (torch.Tensor):
        mid_band_comp_ratio (torch.Tensor):
        mid_band_exp_thresh (torch.Tensor):
        mid_band_exp_ratio (torch.Tensor):
        mid_band_at (torch.Tensor):
        mid_band_rt (torch.Tensor):
        high_shelf_comp_thresh (torch.Tensor):
        high_shelf_comp_ratio (torch.Tensor):
        high_shelf_exp_thresh (torch.Tensor):
        high_shelf_exp_ratio (torch.Tensor):
        high_shelf_at (torch.Tensor):
        high_shelf_rt (torch.Tensor):

    Returns:
        y (torch.Tensor): Filtered signal.
    """
    bs, chs, seq_len = x.size()

    low_cutoff = low_cutoff.view(-1, 1, 1)
    high_cutoff = high_cutoff.view(-1, 1, 1)
    mix = mix.view(-1, 1, 1)

    eff_bs = x.size(0)

    ''' cross over filter '''
    # Low-shelf band (low frequencies)
    low_band = linkwitz_riley_4th_order(x, low_cutoff, sample_rate, filter_type="low_pass")
    # High-shelf band (high frequencies)
    high_band = linkwitz_riley_4th_order(x, high_cutoff, sample_rate, filter_type="high_pass")
    # Mid-band (band-pass)
    mid_band = x - low_band - high_band  # Subtract low and high bands from original signal

    ''' compressor '''
    try:
        x_out_low = low_band * torchcomp.compexp_gain(low_band.sum(axis=1).abs(),
                                            comp_thresh=low_shelf_comp_thresh, \
                                            comp_ratio=low_shelf_comp_ratio, \
                                            exp_thresh=low_shelf_exp_thresh, \
                                            exp_ratio=low_shelf_exp_ratio, \
                                            at=torchcomp.ms2coef(low_shelf_at, sample_rate), \
                                            rt=torchcomp.ms2coef(low_shelf_rt, sample_rate)).unsqueeze(1)
    except:
        x_out_low = low_band
        print('\t!!!failed computing low-band compression!!!')
    try:
        x_out_high = high_band * torchcomp.compexp_gain(high_band.sum(axis=1).abs(),
                                            comp_thresh=high_shelf_comp_thresh, \
                                            comp_ratio=high_shelf_comp_ratio, \
                                            exp_thresh=high_shelf_exp_thresh, \
                                            exp_ratio=high_shelf_exp_ratio, \
                                            at=torchcomp.ms2coef(high_shelf_at, sample_rate), \
                                            rt=torchcomp.ms2coef(high_shelf_rt, sample_rate)).unsqueeze(1)
    except:
        x_out_high = high_band
        print('\t!!!failed computing high-band compression!!!')
    try:
        x_out_mid = mid_band * torchcomp.compexp_gain(mid_band.sum(axis=1).abs(),
                                            comp_thresh=mid_band_comp_thresh, \
                                            comp_ratio=mid_band_comp_ratio, \
                                            exp_thresh=mid_band_exp_thresh, \
                                            exp_ratio=mid_band_exp_ratio, \
                                            at=torchcomp.ms2coef(mid_band_at, sample_rate), \
                                            rt=torchcomp.ms2coef(mid_band_rt, sample_rate)).unsqueeze(1)
    except:
        x_out_mid = mid_band
        print('\t!!!failed computing mid-band compression!!!')
    x_out = x_out_low + x_out_high + x_out_mid

    # parallel computation
    x_out = mix * x_out + (1-mix) * x

    # move channels back
    x_out = x_out.view(bs, chs, seq_len)

    return x_out



class Limiter(Processor):
    def __init__(
        self,
        sample_rate: int,
        min_threshold_db: float = -60.0,
        max_threshold_db: float = 0.0-EPS,
        min_attack_ms: float = 5.0,
        max_attack_ms: float = 100.0,
        min_release_ms: float = 5.0,
        max_release_ms: float = 100.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.process_fn = limiter
        self.param_ranges = {
            "threshold": (min_threshold_db, max_threshold_db),
            "at": (min_attack_ms, max_attack_ms),
            "rt": (min_release_ms, max_release_ms),
        }
        self.num_params = len(self.param_ranges)

def limiter(
    x: torch.Tensor,
    sample_rate: float,
    threshold: float,
    at: float,
    rt: float,
):
    """Limiter.
    Args:
        x (torch.Tensor): Time domain tensor with shape (bs, chs, seq_len)
        sample_rate (float): Audio sample rate.
        threshold (torch.Tensor): Limiter threshold in dB.
        at (torch.Tensor): Attack time.
        rt (torch.Tensor): Release time.

    Returns:
        y (torch.Tensor): Limited signal.
    """
    bs, chs, seq_len = x.size()
    x_sum = x.sum(axis=1).abs()
    x_out = x * torchcomp.limiter_gain(x_sum + EPS,
                                        threshold=threshold - EPS,
                                        at=torchcomp.ms2coef(at, sample_rate),
                                        rt=torchcomp.ms2coef(rt, sample_rate)).unsqueeze(1)
    # move channels back
    x_out = x_out.view(bs, chs, seq_len)
    return x_out

class Panner(Processor):
    def __init__(
        self,
        sample_rate: int,
        min_pan: float = -1.0,
        max_pan: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.process_fn = panner
        self.param_ranges = {
            "pan": (min_pan, max_pan),
            "mix": (0.2, 0.7),
        }
        self.num_params = len(self.param_ranges)

def panner(
    x: torch.Tensor,
    sample_rate: int,
    pan: torch.Tensor,
    mix: torch.Tensor,
):
    """Stereo panner with constant power panning law.
    Args:
        x (torch.Tensor): Input audio tensor with shape (bs, chs, seq_len)
        sample_rate (int): Audio sample rate.
        pan (torch.Tensor): Pan position from -1 (left) to 1 (right) with shape (bs)
        mix (torch.Tensor): Wet/dry mix factor with shape (bs)
    Returns:
        torch.Tensor: Output audio tensor with shape (bs, chs, seq_len)
    """
    bs, chs, seq_len = x.size()

    # Ensure we have at least 2 channels for stereo panning
    if chs == 1:
        # Convert mono to stereo
        x = x.repeat(1, 2, 1)
        chs = 2
    elif chs > 2:
        # For multi-channel, only pan the first two channels
        x_stereo = x[:, :2, :]
        x_rest = x[:, 2:, :]
    else:
        x_stereo = x
        x_rest = None

    pan = pan.view(bs, 1, 1)
    mix = mix.view(bs, 1, 1)

    # Constant power panning law
    # Convert pan from [-1, 1] to [0, π/2]
    pan_angle = (pan + 1.0) * torch.pi / 4.0

    # Calculate left and right gains
    left_gain = torch.cos(pan_angle)
    right_gain = torch.sin(pan_angle)

    # Apply panning to stereo channels
    if chs >= 2:
        left_channel = x_stereo[:, 0:1, :] * left_gain + x_stereo[:, 1:2, :] * (1 - right_gain)
        right_channel = x_stereo[:, 0:1, :] * (1 - left_gain) + x_stereo[:, 1:2, :] * right_gain
        x_panned = torch.cat([left_channel, right_channel], dim=1)
        # Combine with remaining channels if any
        if x_rest is not None:
            x_panned = torch.cat([x_panned, x_rest], dim=1)
    else:
        x_panned = x

    # Parallel computation (wet/dry mix)
    x_out = mix * x_panned + (1 - mix) * x

    return x_out

class Imager(Processor):
    def __init__(
        self,
        sample_rate: int,
        min_width: float = 0.0,
        max_width: float = 2.0,
        min_mix: float = 0.1,
        max_mix: float = 0.9,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.process_fn = imager
        self.param_ranges = {
            "width": (min_width, max_width),
            "mix": (min_mix, max_mix),
        }
        self.num_params = len(self.param_ranges)

def imager(
    x: torch.Tensor,
    sample_rate: int,
    width: torch.Tensor,
    mix: torch.Tensor,
):
    """Stereo width control using Mid-Side processing.
    Args:
        x (torch.Tensor): Input audio tensor with shape (bs, chs, seq_len)
        sample_rate (int): Audio sample rate.
        width (torch.Tensor): Stereo width factor with shape (bs)
                             0.0 = mono, 1.0 = original, >1.0 = wider
        mix (torch.Tensor): Wet/dry mix factor with shape (bs)

    Returns:
        torch.Tensor: Output audio tensor with shape (bs, chs, seq_len)
    """
    bs, chs, seq_len = x.size()

    # Ensure we have at least 2 channels for stereo imaging
    if chs == 1:
        # Convert mono to stereo
        x = x.repeat(1, 2, 1)
        chs = 2
    elif chs > 2:
        # For multi-channel, only process the first two channels
        x_stereo = x[:, :2, :]
        x_rest = x[:, 2:, :]
    else:
        x_stereo = x
        x_rest = None

    width = width.view(bs, 1, 1)
    mix = mix.view(bs, 1, 1)

    # Mid-Side encoding
    left = x_stereo[:, 0:1, :]
    right = x_stereo[:, 1:2, :]
    mid = (left + right) / 2.0  # Sum (mono component)
    side = (left - right) / 2.0  # Difference (stereo component)
    # Apply width control to side signal
    side_processed = side * width
    # Mid-Side decoding
    left_out = mid + side_processed
    right_out = mid - side_processed
    x_imaged = torch.cat([left_out, right_out], dim=1)
    # Combine with remaining channels if any
    if x_rest is not None:
        x_imaged = torch.cat([x_imaged, x_rest], dim=1)
    # Parallel computation (wet/dry mix)
    x_out = mix * x_imaged + (1 - mix) * x
    return x_out

def linkwitz_riley_4th_order(
    x: torch.Tensor,
    cutoff_freq: torch.Tensor,
    sample_rate: float,
    filter_type: str):
    q_factor = torch.ones(cutoff_freq.shape) / torch.sqrt(torch.tensor([2.0]))
    gain_db = torch.zeros(cutoff_freq.shape)
    q_factor = q_factor.to(x.device)
    gain_db = gain_db.to(x.device)

    b, a = biquad(
        gain_db,
        cutoff_freq,
        q_factor,
        sample_rate,
        filter_type
    )

    del gain_db
    del q_factor

    eff_bs = x.size(0)
    # six second order sections
    sos = torch.cat((b, a), dim=-1).unsqueeze(1)
    # apply filter twice to phase difference amounts of 360°
    x = sosfilt_via_fsm(sos, x)
    x_out = sosfilt_via_fsm(sos, x)
    return x_out
