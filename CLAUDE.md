# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chatterbox Streaming is an open-source Text-to-Speech (TTS) system with real-time streaming capabilities. It features voice cloning, emotion exaggeration control, and achieves a real-time factor (RTF) of 0.499 on 4090 GPUs.

## Installation and Setup

```bash
# Development installation
git clone https://github.com/davidbrowne17/chatterbox-streaming.git
pip install -e .

# Or install from PyPI
pip install chatterbox-streaming
```

## Core Architecture

### Main Components

- **TTS Module** (`src/chatterbox/tts.py`): Main `ChatterboxTTS` class for text-to-speech generation
- **Voice Conversion Module** (`src/chatterbox/vc.py`): Voice cloning and conversion functionality
- **Model Components**:
  - `T3`: 0.5B Llama backbone for text processing
  - `S3Gen`: Speech generation with HiFi-GAN vocoder
  - `S3Tokenizer`: Speech tokenization
  - `VoiceEncoder`: Speaker embedding extraction

### Streaming Architecture

The streaming implementation processes text in chunks and yields audio in real-time:

```python
# Basic streaming usage
model = ChatterboxTTS.from_pretrained(device="cuda")
for audio_chunk, metrics in model.generate_stream(text):
    # Process audio_chunk immediately for real-time playback
    pass
```

### Key Classes

- `ChatterboxTTS`: Main TTS interface with `generate()` and `generate_stream()` methods
- `ChatterboxVC`: Voice conversion interface
- Streaming metrics are tracked via dataclasses containing RTF, latency, and chunk information

## Development Commands

### Running Examples

```bash
# Basic TTS streaming example
python example_tts_stream.py

# Voice cloning example
python example_vc_stream.py

# Gradio interfaces for testing
python gradio_tts_app.py
python gradio_vc_app.py
```

### Training and Fine-tuning

```bash
# LoRA fine-tuning (requires 18GB+ VRAM)
python lora.py

# GRPO fine-tuning (requires 12GB+ VRAM)
python grpo.py

# Load and merge trained checkpoints
python loadandmergecheckpoint.py
```

### Voice Conversion

```bash
# Standalone voice conversion
python voice_conversion.py
```

## Key Parameters

### Generation Parameters
- `exaggeration`: Emotion intensity control (0.0-1.0+), default 0.5
- `cfg_weight`: Classifier-free guidance weight (0.0-1.0), default 0.5
- `temperature`: Sampling randomness (0.1-1.0)
- `chunk_size`: Speech tokens per chunk for streaming (default: 50)

### Performance Tuning
- Smaller `chunk_size` = lower latency but more overhead
- Lower `cfg_weight` (~0.3) for fast-paced speech
- Higher `exaggeration` (~0.7+) for dramatic speech

## Project Structure

```
src/chatterbox/
├── tts.py              # Main TTS class and streaming logic
├── vc.py               # Voice conversion functionality
├── models/
│   ├── t3/            # Llama backbone and T3 model
│   ├── s3gen/         # Speech generation (HiFi-GAN, flow matching)
│   ├── s3tokenizer/   # Speech tokenization
│   ├── tokenizers/    # Text tokenization
│   └── voice_encoder/ # Speaker embedding extraction
└── __init__.py
```

## Performance Metrics

Target performance on 4090 GPU:
- Latency to first chunk: ~0.472s
- Real-Time Factor (RTF): 0.499 (target < 1.0)
- Sample rate: 24kHz

## Dependencies

Key dependencies managed via `pyproject.toml`:
- PyTorch 2.6.0 with torchaudio
- Transformers 4.46.3
- librosa, resampy for audio processing
- resemble-perth for watermarking
- sounddevice for real-time audio playback

## Notes

- No test suite currently exists in the repository
- Training requires CUDA GPU with 12-18GB+ VRAM depending on method
- All generated audio includes Perth watermarks for responsible AI
- Model weights are loaded from Hugging Face Hub (`ResembleAI/chatterbox`)