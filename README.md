# NetImage

A from-scratch SDXL (Stable Diffusion XL) text-to-image pipeline built entirely in C# using ONNX Runtime with DirectML GPU acceleration. No Python runtime needed at inference time.

![Example output — "a photo of an astronaut riding a horse on mars"](docs/example.png)

## Features

- Full SDXL text-to-image generation pipeline in .NET
- GPU-accelerated inference via DirectML (works with NVIDIA, AMD, and Intel GPUs on Windows)
- BPE tokenization with dual CLIP text encoders
- Euler discrete noise scheduler
- Classifier-free guidance support
- Configurable resolution, steps, guidance scale, and seed

## Requirements

- .NET 10 SDK
- Windows 10/11 (DirectML requirement)
- GPU with DirectX 12 support (any modern NVIDIA, AMD, or Intel GPU)
- SDXL model exported to ONNX format (see [Model Setup](#model-setup))

## Quick Start

```bash
# Build
dotnet build

# Generate an image
dotnet run --project Net-Image -- \
  --model ./sdxl-onnx \
  --prompt "a photo of an astronaut riding a horse on mars" \
  --output result.png
```

## CLI Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | *(required)* | Path to SDXL ONNX model directory |
| `--prompt` | `-p` | *(required)* | Text prompt for image generation |
| `--negative-prompt` | `-n` | `""` | Negative prompt |
| `--output` | `-o` | `output.png` | Output file path |
| `--steps` | `-s` | `30` | Number of denoising steps |
| `--guidance-scale` | `-g` | `7.5` | Classifier-free guidance scale |
| `--seed` | | `42` | Random seed for reproducibility |
| `--width` | | `1024` | Output width in pixels |
| `--height` | | `1024` | Output height in pixels |

## Model Setup

The pipeline requires an SDXL model converted to ONNX format. You'll need Python for the one-time conversion:

```bash
pip install optimum[onnxruntime]
optimum-cli export onnx --model stabilityai/stable-diffusion-xl-base-1.0 --task text-to-image ./sdxl-onnx
```

This produces the following directory structure:

```
sdxl-onnx/
├── text_encoder/model.onnx
├── text_encoder_2/model.onnx
├── tokenizer/vocab.json, merges.txt
├── tokenizer_2/vocab.json, merges.txt
├── unet/model.onnx
└── vae_decoder/model.onnx
```

## Architecture

```
Program.cs (CLI)
    |
    v
StableDiffusionPipeline        orchestrates the full generation loop
    |
    |-- ClipTokenizer (x2)     tokenizes prompt text via BPE
    |-- TextEncoder (x2)       token IDs -> text embeddings (ONNX Runtime)
    |-- EulerDiscreteScheduler manages noise schedule and timesteps
    |-- UNet                   predicts noise at each step (ONNX Runtime)
    +-- VaeDecoder             latent -> RGB pixels (ONNX Runtime)
            |
            v
        ImageSharp -> output PNG
```

### Pipeline Flow

1. **Tokenize** the prompt with two CLIP tokenizers (77 tokens max each)
2. **Encode** with two text encoders, concatenate hidden states (768 + 1280 = 2048 dimensions), extract pooled output from encoder 2
3. **Generate** random latent noise (1x4x128x128 for 1024x1024 output)
4. **Denoise** loop — for each scheduler timestep:
   - Duplicate latent for classifier-free guidance (unconditional + conditional)
   - Run U-Net with text embeddings, timestep, and micro-conditioning
   - Apply guidance: `noise_pred = uncond + scale * (cond - uncond)`
   - Scheduler step to update latent
5. **Decode** final latent with VAE decoder
6. **Save** as PNG

## Tips

- **Higher guidance scale** (8-12) makes the model follow the prompt more closely
- **More steps** (40-50) produces finer detail at the cost of longer generation time
- **Different seeds** produce completely different compositions
- SDXL works best at **1024x1024** resolution

## License

MIT
