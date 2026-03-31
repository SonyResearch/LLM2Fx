import os
import torch
import ast
import pandas as pd
import torchaudio
import librosa
import numpy as np
import random
from torch.utils.data import Dataset
from llm4mp.common_utils.fx_utils import load_fx_config
from llm4mp.models.llm2fx2.llm.model_setup import setup_tokenizer
from transformers.utils import get_json_schema
from llm4mp.common_utils.fx_utils.ndsp_utils.tool_calling_fn import ndsp_fx_tools
from llm4mp.common_utils.audio_utils.audio_io import get_dry_audio, get_wet_audio
from llm4mp.preprocessing.fx_processor.fx_sampling_ndsp import sample_fx_parameters

class AudioFXDataset(Dataset):
    def __init__(self, db, audio_dir, model_path, cache_dir, fx_config_type="ndsp", target_duration=10, split="train", use_cot=True, use_chat=True, online_sampling=False):
        self.metadata = db
        self.audio_dir = audio_dir
        self.target_duration = target_duration
        self.fx_config_type = fx_config_type
        self.fx_config = load_fx_config(fx_config_type)
        self.tokenizer = setup_tokenizer(model_path, cache_dir)
        self.start_of_tool = "<tool_call>"
        self.end_of_tool = "</tool_call>"
        self.start_to_think = "<think>"
        self.end_to_think = "</think>"
        self.start_of_audio = "<|vision_start|>" # we use Qwen Vision Start token for audio, for bypass vocabulary expansion
        self.end_of_msg = '<|im_end|>'
        self.split = split
        self.use_cot = use_cot
        self.use_chat = use_chat
        self.online_sampling = online_sampling
        self.ndsp_fx_tools = [get_json_schema(v) for k,v in ndsp_fx_tools.items()]

    def __len__(self):
        return len(self.metadata)

    def get_dry_wet_audio_random_pair(self, row):
        _id = row['id']
        filename = row['filename']
        fname = row['fname']
        random_chunk_idx = random.choice(range(3))
        dry_path = f"{self.audio_dir}/{fname}/{filename}/{_id}/dry_{random_chunk_idx}.flac"
        if not os.path.exists(dry_path):
            dry_path = f"{self.audio_dir}/{fname}/{filename}/{_id}/dry_0.flac"
        audio_dry, _ = librosa.load(dry_path, sr=None, mono=False)
        fx_pool = self.fx_config['fx_list'].copy()
        num_of_fx = random.choices(list(range(1, 10)), weights=[(1/i)**2 for i in range(1, 10)], k=1)[0]
        sampled_fx_chain = random.sample(fx_pool, num_of_fx)
        tools = []
        for fx_name in sampled_fx_chain:
            fx_param_range = self.fx_config['fx_param_ranges'][fx_name]
            random_fine_grained = random.random() < 0.5
            samples = sample_fx_parameters(fx_param_range, num_samples=1, fine_grained=random_fine_grained)
            tools.append({"name": fx_name, "arguments": samples})
        audio_wet = get_wet_audio(audio_dry, tools, self.fx_config)
        audio_wet = np.array(audio_wet).astype(np.float32)
        return audio_dry, audio_wet, tools

    def get_dry_wet_audio_pair(self, row):
        _id = row['id']
        filename = row['filename']
        fname = row['fname']
        tools = ast.literal_eval(row['tools'])
        random_chunk_idx = random.choice(range(3))
        dry_path = f"{self.audio_dir}/{fname}/{filename}/{_id}/dry_{random_chunk_idx}.flac"
        wet_path = f"{self.audio_dir}/{fname}/{filename}/{_id}/wet_{random_chunk_idx}.flac"
        if not os.path.exists(dry_path) or not os.path.exists(wet_path):
            dry_path = f"{self.audio_dir}/{fname}/{filename}/{_id}/dry_0.flac"
            wet_path = f"{self.audio_dir}/{fname}/{filename}/{_id}/wet_0.flac"
        audio_dry, _ = librosa.load(dry_path, sr=None, mono=False)
        audio_wet, _ = librosa.load(wet_path, sr=None, mono=False)
        return audio_dry, audio_wet, tools

    def __getitem__(self, idx):
        row = self.metadata[idx]
        if self.online_sampling:
            audio_dry, audio_wet, tools = self.get_dry_wet_audio_random_pair(row)
        else:
            audio_dry, audio_wet, tools = self.get_dry_wet_audio_pair(row)
        # load conversational dataset
        conversations = row['conversations']
        user_query = conversations[0]['content']
        assistant_response = conversations[1]['content']
        chain_of_thought = f"{row['chain_of_thought']}\n{self.end_to_think}"
        target_tools = []
        for tool_info in tools:
            target_tools.append(f"{self.start_of_tool}\n{tool_info}\n{self.end_of_tool}")
        target_tools = "".join(target_tools)
        if self.use_cot:
            output_text = f"{chain_of_thought}\n{target_tools}\n{assistant_response}\n{self.end_of_msg}"
        else:
            output_text = f"\n{target_tools}\n{assistant_response}\n{self.end_of_msg}"
        if self.use_chat:
            input_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{user_query} \n {self.start_of_audio}"}],
                tools=self.ndsp_fx_tools,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=True,
            )
        else:
            input_text = f"{self.start_of_audio}"
            output_text = f"{target_tools}"
        return {
            "input_text": input_text,
            "output_text": output_text,
            "audio_dry": torch.from_numpy(audio_dry),
            "audio_wet": torch.from_numpy(audio_wet),
        }
