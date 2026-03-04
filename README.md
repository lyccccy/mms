# MMS: Music Multimodal Semantic Codec

A neural audio codec that extends [Descript Audio Codec (DAC)](https://github.com/descriptinc/descript-audio-codec) with semantic conditioning and knowledge distillation from [Demucs](https://github.com/facebookresearch/demucs).

## Overview

MMS (Music Multimodal Semantic codec) combines the compression efficiency of DAC with the semantic understanding of Demucs to produce higher-quality audio codec representations for music. By distilling knowledge from Demucs — a state-of-the-art music source separation model — into the DAC codec, MMS learns perceptually meaningful latent representations that better capture musical structure.

### Key Features

- **Stereo audio codec** at 44.1 kHz with 9 residual vector quantization (RVQ) codebooks
- **Semantic conditioning**: fuse Demucs encoder features into DAC latent space before quantization
- **Knowledge distillation**: align DAC's intermediate representations with Demucs semantic features
- **Multiple operating modes**: `dac`, `distill`, `concat`, `add`
- **GAN-based training**: multi-period, multi-scale, and multi-resolution discriminators

## Architecture

```
Input Audio (stereo, 44.1kHz)
        │
        ├──────────────────────┐
        ▼                      ▼
  DAC Encoder            Demucs Encoder
  (2→512 channels)       (semantic teacher)
        │                      │
        └──────┬───────────────┘
               │  (concat / add / distill)
               ▼
  Residual Vector Quantizer
  (9 codebooks, size 1024)
               │
               ▼
  DAC Decoder → Reconstructed Audio
```

### Operating Modes

| Mode | Description |
|------|-------------|
| `dac` | Standard DAC training without semantic conditioning |
| `distill` | Distillation loss only — align DAC features with Demucs features |
| `concat` | Concatenate Demucs features to DAC latent before quantization |
| `add` | Add Demucs features to DAC latent before quantization |

## Dependencies

Install the following packages before running:

```bash
pip install torch torchaudio
pip install audiotools       # Descript audiotools
pip install argbind
pip install accelerate       # for multi-GPU training
pip install librosa numpy einops julius tqdm safetensors tensorboard
```

> **Note:** No `requirements.txt` is currently provided. The above list reflects imports found across the codebase.

## Project Structure

```
mms/
├── mss.py              # MSS model: DAC + optional Demucs semantic fusion
├── singlegpu.py        # Single-GPU training script
├── ddptrain.py         # Multi-GPU training script (via Hugging Face Accelerate)
├── dacdataset.py       # Dataset and dataloader utilities
│
├── model/              # Project-specific model components
│   ├── dac.py          # Stereo DAC (2-channel encoder/decoder)
│   ├── demucs.py       # Demucs teacher model
│   ├── base.py         # DACFile and CodecMixin (compress/decompress)
│   ├── utlis.py        # Utilities (center_trim, etc.)
│   └── diffq/          # Differentiable quantization
│
└── dac/                # Descript Audio Codec package
    ├── __main__.py     # CLI: encode, decode, download
    ├── model/
    │   ├── dac.py      # Reference DAC (mono)
    │   └── discriminator.py  # MPD, MSD, MRD discriminators
    ├── nn/
    │   ├── layers.py
    │   ├── loss.py     # Mel, STFT, GAN losses
    │   └── quantize.py
    └── utils/
        ├── encode.py
        └── decode.py
```

## Usage

### Encode Audio

```bash
python -m dac encode \
  --input path/to/audio.wav \
  --output path/to/output/ \
  --weights_path path/to/weights.pth \
  --model_type 44khz \
  --device cuda
```

### Decode Audio

```bash
python -m dac decode \
  --input path/to/output.dac \
  --output path/to/decoded/ \
  --weights_path path/to/weights.pth \
  --device cuda
```

### Download Pretrained DAC Weights

```bash
python -m dac download --model_type 44khz --model_bitrate 8kbps
```

### Training

**Single GPU:**

```bash
python singlegpu.py
```

**Multi-GPU (via Accelerate):**

```bash
accelerate launch ddptrain.py
```

> Training paths (filelists, teacher weights, checkpoint directories) are currently hardcoded in the training scripts. Edit `singlegpu.py` / `ddptrain.py` to point to your data and model paths before running.

## Training Details

| Parameter | Value |
|-----------|-------|
| Sample rate | 44,100 Hz |
| Channels | 2 (stereo) |
| Hop length | 2048 |
| Codebooks | 9 |
| Codebook size | 1024 |
| Optimizer | AdamW (lr=1e-4, β=[0.8, 0.99]) |
| LR scheduler | ExponentialLR (γ=0.999996) |

**Loss components:**

- Mel spectrogram loss (weight: 100×)
- Multi-scale STFT loss
- Waveform L1 loss
- GAN generator + feature matching loss
- VQ commitment + codebook loss
- Semantic distillation loss (when `mode=distill`)

## Based On

- **[Descript Audio Codec (DAC)](https://github.com/descriptinc/descript-audio-codec)** — high-fidelity neural audio codec
- **[Demucs](https://github.com/facebookresearch/demucs)** — music source separation model used as semantic teacher

## License

Please refer to the licenses of the upstream projects:
- DAC: [MIT License](https://github.com/descriptinc/descript-audio-codec/blob/main/LICENSE)
- Demucs: [MIT License](https://github.com/facebookresearch/demucs/blob/main/LICENSE)
