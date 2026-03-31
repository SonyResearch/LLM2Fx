import torch
import torch.nn as nn
from .modules import ParametricEQ, Distortion, Compressor, Gain, NoiseShapedReverb
# from .custom_modules import Distortion, Multiband_Compressor, Limiter

class Random_FX_Augmentation(nn.Module):
    """
    A module that applies random audio effects to input signals.

    This class creates a chain of audio effects processors and applies them
    with configurable probabilities. Each effect can be independently enabled
    or disabled for each sample in a batch.

    Args:
        sample_rate (int): The sample rate of the audio signals.
        tgt_fx_names (list): List of effect names to include in the processing chain.
                            Default: ['eq', 'comp', 'reverb']
    """
    def __init__(self, sample_rate,
                 tgt_fx_names=['eq', 'comp', 'reverb']):
        super(Random_FX_Augmentation, self).__init__()
        self.sample_rate = sample_rate
        self.tgt_fx_names = tgt_fx_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Probability of applying each effect
        self.fx_prob = {
            'eq': 1.0,
            'distortion': 0.3,
            'comp': 0.8,
            'multiband_comp': 0.8,
            'gain': 0.85,
            'imager': 0.6,
            'limiter': 1.0,
            'reverb': 1.0
        }
        # Initialize effect processors
        self.fx_processors = {}
        for fx_name in tgt_fx_names:
            if fx_name == 'eq':
                fx_module = ParametricEQ(
                    sample_rate=sample_rate,
                    min_gain_db=-40.0,
                    max_gain_db=40.0,
                    min_q_factor=0,
                    max_q_factor=10.0
                )
            elif fx_name == 'distortion':
                fx_module = Distortion(
                    sample_rate=sample_rate,
                    min_gain_db=0.0,
                    max_gain_db=4.0
                )
            elif fx_name == 'comp':
                fx_module = Compressor(sample_rate=sample_rate)
            elif fx_name == 'reverb':
                fx_module = NoiseShapedReverb(
                    sample_rate=sample_rate,
                    min_band_gain=0.0,
                    max_band_gain=2.0,
                    min_band_decay=0.0,
                    max_band_decay=1.0,
                    min_mix=0.0,
                    max_mix=1.0,
                )
            else:
                raise AssertionError(f"Current effect name ({fx_name}) not found")

            self.fx_processors[fx_name] = fx_module
        # Calculate total number of parameters across all effects
        total_num_param = sum([self.fx_processors[fx_name].num_params for fx_name in self.fx_processors])
        self.total_num_param = total_num_param

    def forward(self, x, use_mask=True, return_denorm_param_dicts=True):
        """
        Apply the chain of audio effects to the input signal.
        Args:
            x (torch.Tensor): Input audio tensor with shape (batch_size, channels, samples)
            use_mask (bool, optional): Whether to use a mask to decide whether to apply the effect.
                                       Default: True
        Returns:
            torch.Tensor: Processed audio with the same shape as input
            dict: Dictionary of denormalized parameters for each effect
        """
        x = x.to(self.device)
        batch_size = x.shape[0]
        if use_mask:
            random_mask = self.random_mask_generator(x.shape[0])

        batch_denorm_param_dicts = [{fx_name: None for fx_name in self.tgt_fx_names}] * batch_size
        for fx_name in self.tgt_fx_names:
            current_processor = self.fx_processors[fx_name]
            current_num_params = current_processor.num_params
            rand_param = torch.rand((batch_size, current_num_params)).to(self.device)
            processed_audio, denorm_param_dict = current_processor.process_normalized(x.to(self.device), rand_param)
            cur_mask = random_mask[fx_name]
            x = processed_audio * cur_mask + x * ~cur_mask
            if return_denorm_param_dicts:
                for i in range(batch_size):
                    if cur_mask[i].flatten():
                        # !important: we apply same effect to all samples in the batch
                        batch_denorm_param_dicts[i][fx_name] = {k:v[i].detach().cpu().item() for k,v in denorm_param_dict.items()}
                return x, batch_denorm_param_dicts
        return x

    def random_mask_generator(self, batch_size, repeat=1):
        mask = {}
        for cur_fx in self.tgt_fx_names:
            mask[cur_fx] = self.fx_prob[cur_fx] > torch.rand(batch_size).view(-1, 1, 1)
            if repeat>1:
                mask[cur_fx] = mask[cur_fx].repeat(repeat, 1, 1)
            mask[cur_fx] = mask[cur_fx].to(self.device)
        return mask
