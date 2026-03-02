namespace Net_Image.Pipeline;

public class PipelineConfig
{
    public required string ModelDirectory { get; init; }
    public required string Prompt { get; init; }
    public string NegativePrompt { get; init; } = "";
    public int NumInferenceSteps { get; init; } = 30;
    public float GuidanceScale { get; init; } = 7.5f;
    public int Seed { get; init; } = 42;
    public string OutputPath { get; init; } = "output.png";
    public int Width { get; init; } = 1024;
    public int Height { get; init; } = 1024;

    public string TextEncoderPath => Path.Combine(ModelDirectory, "text_encoder", "model.onnx");
    public string TextEncoder2Path => Path.Combine(ModelDirectory, "text_encoder_2", "model.onnx");
    public string UNetPath => Path.Combine(ModelDirectory, "unet", "model.onnx");
    public string VaeDecoderPath => Path.Combine(ModelDirectory, "vae_decoder", "model.onnx");
    public string Tokenizer1VocabPath => Path.Combine(ModelDirectory, "tokenizer", "vocab.json");
    public string Tokenizer1MergesPath => Path.Combine(ModelDirectory, "tokenizer", "merges.txt");
    public string Tokenizer2VocabPath => Path.Combine(ModelDirectory, "tokenizer_2", "vocab.json");
    public string Tokenizer2MergesPath => Path.Combine(ModelDirectory, "tokenizer_2", "merges.txt");
}
