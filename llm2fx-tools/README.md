# LLM2Fx-Tools: Tool Calling For Music Post-Production

- **Paper**: [arXiv:2512.01559](https://arxiv.org/abs/2512.01559)
- **Demo**: [llm2fx-tools-demo](https://seungheondoh.github.io/llm2fx-tools-demo/)
- **Dataset**: (coming soon)

A multimodal framework that fine-tunes LLMs to generate executable sequences of audio effect tool calls for music post-production. Given a pair of dry and wet audio signals, the model predicts the effect chain (type, order, and parameters) as structured tool calls via Supervised Fine-Tuning (SFT).

## Architecture

The system is composed of three trainable components:

```
[Dry Audio] ──► Audio Encoder ──►┐
                                  ├──► Projection Layer ──► LLM (LoRA SFT) ──► Tool Call Sequence
[Wet Audio] ──► Audio Encoder ──►┘
```

- **Audio Encoder**: Encodes dry/wet audio pairs into embeddings (`fxenc_plusplus`, `fxenc`, `clap`, or `stito_encoder`)
- **Projection Layer**: MLP or cross-attention adapter that maps audio embeddings to the LLM's hidden dimension
- **LLM**: Qwen3 / Llama / Mistral fine-tuned with LoRA; generates chain-of-thought reasoning followed by structured tool calls

**Supported effects (DASP backend):**
`gain`, `distortion`, `compressor_expander`, `limiter`, `panner`, `imager`, `reverb`, `parametric_eq`

**Supported effects (NDSP/Pedalboard backend):**
`gain`, `distortion`, `reverb`, `compressor`, `delay`, `limiter`, `equalizer`, `panner`, `stereo_widener`

## Installation

```bash
uv venv .venv
source .venv/bin/activate
uv add torch torchvision torchaudio lightning
uv pip install -e .
```

## Training

Training is driven by a YAML config file using PyTorch Lightning with FSDP or DDP support.

```bash
# Single GPU
python train.py --config configs/your_config.yaml

# Multi-GPU with FSDP
torchrun --nproc_per_node=8 train.py --config configs/your_config.yaml
```

Key config options:

| Option | Values | Description |
|---|---|---|
| `model_type` | `qwen3`, `llama`, `mistral` | Base LLM backbone |
| `audio_encoder_type` | `fxenc_plusplus`, `fxenc`, `clap` | Audio encoder |
| `projector_type` | `mlp`, `adapter` | Audio-to-LLM projection |
| `apply_lora` | `true` / `false` | LoRA fine-tuning |
| `loss_type` | `ce`, `ce_ntl` | Cross-entropy or CE + NT-Loss |
| `use_cot` | `true` / `false` | Chain-of-thought in training output |
| `online_sampling` | `true` / `false` | Apply FX chains at runtime vs. use pre-computed pairs |
| `dry_audio_dropout` | `0.0` – `1.0` | Probability to mask dry audio during training |

## Dataset Format

The dataset is loaded from Hugging Face datasets. Each sample contains audio file paths and conversation metadata:

```
audio_dir/
└── {fname}/
    └── {filename}/
        └── {id}/
            ├── dry_0.flac
            ├── wet_0.flac
            └── ...
```

Each metadata record includes:
- `conversations`: user query / assistant response pairs
- `tools`: list of audio effect tool schemas
- `chain_of_thought`: reasoning text for CoT training

## Citation

```bibtex
@inproceedings{doh2026llm2fx,
  title={LLM2Fx-Tools: Tool Calling For Music Post-Production},
  author={Doh*, Seungheon and Koo*, Junghyun and Mart{\'\i}nez-Ram{\'\i}rez, Marco A and Choi, Woosung and Liao, Wei-Hsiang and Wu, Qiyu and Nam, Juhan and Mitsufuji, Yuki},
  note={* Equal contribution},
  booktitle={The Thirteenth International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
