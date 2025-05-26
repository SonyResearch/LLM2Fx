# LLM2Fx: Large Language Models for Audio Effect Parameter Prediction

A research project that leverages Large Language Models (LLMs) to intelligently predict audio effect parameters for music production. This system bridges the gap between natural language descriptions and technical audio processing parameters.

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended)
- FFmpeg

### Setup

```
git clone
conda create -n llm2fx python=3.10
conda activate llm2fx
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e .
```


### Text2FX (LLM predict Fx Parmas)
- Purpose: Translates natural language descriptions into precise audio effect parameters
- Input: Text descriptions (e.g., "make it sound warm and spacious")
- Output: JSON-formatted audio effect parameters (EQ, reverb, etc.)
- Use Case: Allows producers to describe desired sound characteristics in plain English


```bash
python llm2fx/inference/text2fx/run_llm.py
    --eq_type reverb
    --timbre_word underwater
    --inst_type guitar
    --use_incontext True
```
Parameters:
- `--fx_type`: Effect type (`eq`, `reverb`)
- `--model_name`: LLM model (`llama3_1b`, `llama3_3b`, `llama3_8b`, `mistral_7b`)
- `--inst_type`: Instrument type
- `--use_incontext`: Use in-context information or not (if not, it is zero-shot case)
- `--output_path`: Directory for output files
- `--device`: Compute device
```
