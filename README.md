# LLM for Music Post-Production

![LLM for Music Post-Production](https://i.imgur.com/F9buZOL.png)

This repository contains the code and resources for two research papers on applying Large Language Models to audio effect parameter prediction in music production.

---

## Paper 1: LLM2Fx — Text-to-Parameter

**Can Large Language Models Predict Audio Effects Parameters from Natural Language?**

[![arXiv](https://img.shields.io/badge/arXiv-2505.20770-b31b1b.svg)](https://arxiv.org/abs/2505.20770)
[![Demo](https://img.shields.io/badge/Demo-llm2fx--demo-blue)](https://seungheondoh.github.io/llm2fx-demo/)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/collections/seungheondoh/llm2fx-6821b961b982fe1eab1b00bf)

### Overview

LLM2Fx investigates whether LLMs can translate natural language descriptions into audio effect parameters (EQ, reverb) without task-specific training. We show that off-the-shelf LLMs can perform this **Text-to-Parameter** task in a zero-shot manner, and propose three in-context learning strategies — audio DSP features, DSP function code, and few-shot examples — that further boost performance.

- **Zero-shot Text-to-Parameter**: LLMs can generate audio effect parameters directly from text without fine-tuning
- **In-context learning strategies**: DSP feature injection, DSP function code, and few-shot examples for improved accuracy

---

## Paper 2: LLM2Fx-Tools — Audio-to-Parameter via Tool Calling

**LLM2Fx-Tools: Tool Calling For Music Post-Production**

[![arXiv](https://img.shields.io/badge/arXiv-2512.01559-b31b1b.svg)](https://arxiv.org/abs/2512.01559)
[![Demo](https://img.shields.io/badge/Demo-llm2fx--demo-blue)](https://seungheondoh.github.io/llm2fx-tools-demo/)
[![Dataset(soon)](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](#)

### Overview

LLM2Fx-Tools extends the LLM2Fx paradigm to **Audio-to-Parameter** prediction using an LLM tool-calling framework. Given a pair of unprocessed and processed audio signals, the model generates an executable sequence of audio effect modules (tool calls) along with their parameters.

- **Tool-calling framework for audio effects**: LLMs generate structured, executable sequences of audio effect module calls
- **SFT on tool sequences**: LLM fine-tuned to predict effect type, ordering, and parameters autoregressively

---

## Citation

If you use this work, please cite the relevant paper(s):

```bibtex
@article{doh2025llm2fx,
  title     = {Can Large Language Models Predict Audio Effects Parameters from Natural Language?},
  author    = {Doh, Seungheon and Koo, Junghyun and Mart{\'\i}nez-Ram{\'\i}rez, Marco A. and Liao, Wei-Hsiang and Nam, Juhan and Mitsufuji, Yuki},
  journal   = {arXiv preprint arXiv:2505.20770},
  year      = {2025}
}

@article{doh2024llm2fxtools,
  title     = {LLM2Fx-Tools: Tool Calling For Music Post-Production},
  author    = {Doh, Seungheon and Koo, Junghyun and Mart{\'\i}nez-Ram{\'\i}rez, Marco A. and Choi, Woosung and Liao, Wei-Hsiang and Wu, Qiyu and Nam, Juhan and Mitsufuji, Yuki},
  journal   = {arXiv preprint arXiv:2512.01559},
  year      = {2024}
}
```
