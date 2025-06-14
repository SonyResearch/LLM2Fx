# LLM2Fx: Large Language Models for Audio Effect Parameter Prediction

- **Paper**: [Arxiv](#)
- **Demo**: [llm2fx-demo](https://seungheondoh.github.io/llm2fx-demo/)
- **Dataset**: [huggingface](https://huggingface.co/collections/seungheondoh/llm2fx-6821b961b982fe1eab1b00bf)
- **Re-Synthesize SocialFX**: [fx-processor](https://github.com/seungheondoh/fx-processor)


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
    --model_name mistral_7b
    --eq_type reverb
    --timbre_word church
    --inst_type guitar
    --use_incontext True
```

```
Parameters:
- `--fx_type`: Effect type (`eq`, `reverb`)
- `--model_name`: LLM model (`llama3_1b`, `llama3_3b`, `llama3_8b`, `mistral_7b`)
- `--inst_type`: Instrument type
- `--use_incontext`: Use in-context information or not (if not, it is zero-shot case)
- `--output_path`: Directory for output files
- `--device`: Compute device
```

### LLM Output

You can find example outputs in the `./outputs` directory, which contains the results of running the inference process. These examples demonstrate how the model generates audio effect parameters based on different input combinations of instrument types and timbre descriptions. Each output includes both the technical parameters and a human-readable explanation of the reasoning behind the parameter choices.

### Audio Examples

You can listen to the generated audio examples in the `./outputs/text2fx/{model_name}/{fx_type}/{inst_type}/{timbre_word}/audio` directory.

| Instrument | Timbre | Audio Example |
|------------|--------|---------------|
| Guitar | Church | [![Audio Example](https://img.shields.io/badge/🎵-Play%20Audio-blue)](./outputs/text2fx/mistral_7b/reverb/guitar/church/audio/582cf088-4941-4453-9d90-2c75dc76bbd5.mp3) |


```bash
{
    "reverb": {
        "band0_gain": 0.0,
        "band1_gain": 0.1,
        "band2_gain": 0.2,
        "band3_gain": 0.3,
        "band4_gain": 0.4,
        "band5_gain": 0.5,
        "band6_gain": 0.6,
        "band7_gain": 0.7,
        "band8_gain": 0.8,
        "band9_gain": 0.9,
        "band10_gain": 1.0,
        "band11_gain": 0.9,
        "band0_decay": 3.0,
        "band1_decay": 2.5,
        "band2_decay": 2.0,
        "band3_decay": 1.5,
        "band4_decay": 1.2,
        "band5_decay": 1.0,
        "band6_decay": 0.8,
        "band7_decay": 0.6,
        "band8_decay": 0.4,
        "band9_decay": 0.3,
        "band10_decay": 0.2,
        "band11_decay": 0.1,
        "mix": 0.7
    },
    "reason": "A church-like guitar sound is often characterized by a warm, spacious, and reverberant tone.
    This design boosts the gain in all frequency bands to achieve a warm and full sound, while using longer decay times to create a sense of space and ambiance.
    The mix level is set to 0.7 to ensure a balanced level between the dry and wet signals."
}
```

### Embedding Extraction for Evaluation

We use embedding distribution as an evaluation metric based on the fx encoder. For this purpose, we provide embedding extractor code and evaluation metrics (`./evaluation/text2fx/llm_eval.py`).

```bash
# embedding extraction code
python llm2fx/evaluation/embeds/fx_encoder.py --target_dir outputs/text2fx --output_dir outputs/embeds --audio_extension wav
```

