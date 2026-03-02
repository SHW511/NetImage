# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
dotnet build          # Build the solution
dotnet run --project Net-Image -- --model <path-to-sdxl-onnx> --prompt "your prompt" --output result.png
```

No test framework is configured yet.

## Architecture

- **Solution**: `NetImage.slnx` (VS 2022 slnx format) containing a single console application
- **Project**: `Net-Image/` — .NET 10 console app (`net10.0`), C# with nullable reference types and implicit usings enabled
- **Entry point**: `Net-Image/Program.cs`
- **Root namespace**: `Net_Image` (underscore replaces the hyphen in the project name)
- **GPU acceleration**: DirectML via `Microsoft.ML.OnnxRuntime.DirectML` (works with any GPU on Windows)

## Project Structure

```
Net-Image/
├── Program.cs                          # CLI entry point, argument parsing
├── Pipeline/
│   ├── StableDiffusionPipeline.cs      # Main pipeline orchestrator
│   └── PipelineConfig.cs              # Model paths, inference params
├── Tokenizer/
│   └── ClipTokenizer.cs               # BPE tokenizer (vocab.json + merges.txt)
├── Inference/
│   ├── TextEncoder.cs                 # ONNX session for CLIP text encoding
│   ├── UNetInference.cs               # ONNX session for U-Net denoising
│   └── VaeDecoder.cs                  # ONNX session for VAE decoding
├── Schedulers/
│   ├── IScheduler.cs                  # Scheduler interface
│   └── EulerDiscreteScheduler.cs      # Euler discrete noise scheduler
└── Utils/
    ├── TensorHelper.cs                # Tensor creation and math operations
    └── ImageHelper.cs                 # Float tensor to PNG via ImageSharp
```

## Key Dependencies

- `Microsoft.ML.OnnxRuntime.DirectML` — ONNX inference with DirectML GPU acceleration
- `SixLabors.ImageSharp` — Image output (PNG)

## SDXL Pipeline Flow

1. Tokenize prompt with two CLIP tokenizers (77-token max each)
2. Encode with two text encoders, concatenate hidden states (768+1280=2048 dim)
3. Generate random latent noise (1x4x128x128 for 1024x1024 output)
4. Denoise loop with classifier-free guidance via U-Net
5. Decode latents with VAE
6. Save as PNG
