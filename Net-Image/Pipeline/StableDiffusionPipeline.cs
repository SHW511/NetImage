using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Net_Image.Inference;
using Net_Image.Schedulers;
using Net_Image.Tokenizer;
using Net_Image.Utils;

namespace Net_Image.Pipeline;

public sealed class StableDiffusionPipeline : IDisposable
{
    private readonly PipelineConfig _config;
    private readonly ClipTokenizer _tokenizer1;
    private readonly ClipTokenizer _tokenizer2;
    private readonly TextEncoder _textEncoder1;
    private readonly TextEncoder _textEncoder2;
    private readonly UNetInference _unet;
    private readonly VaeDecoder _vaeDecoder;
    private readonly IScheduler _scheduler;

    public StableDiffusionPipeline(PipelineConfig config)
    {
        _config = config;

        Console.WriteLine("Loading tokenizers...");
        _tokenizer1 = new ClipTokenizer(config.Tokenizer1VocabPath, config.Tokenizer1MergesPath);
        _tokenizer2 = new ClipTokenizer(config.Tokenizer2VocabPath, config.Tokenizer2MergesPath);

        var sessionOptions = CreateSessionOptions();

        Console.WriteLine("Loading text encoder 1...");
        _textEncoder1 = new TextEncoder(config.TextEncoderPath, sessionOptions, hiddenSize: 768);

        Console.WriteLine("Loading text encoder 2...");
        _textEncoder2 = new TextEncoder(config.TextEncoder2Path, sessionOptions, hiddenSize: 1280);

        Console.WriteLine("Loading U-Net...");
        _unet = new UNetInference(config.UNetPath, sessionOptions);

        Console.WriteLine("Loading VAE decoder...");
        _vaeDecoder = new VaeDecoder(config.VaeDecoderPath, sessionOptions);

        _scheduler = new EulerDiscreteScheduler();
    }

    public void Generate()
    {
        var totalSw = Stopwatch.StartNew();

        // 1. Tokenize
        Console.WriteLine("Tokenizing prompt...");
        var tokens1 = _tokenizer1.Tokenize(_config.Prompt);
        var tokens2 = _tokenizer2.Tokenize(_config.Prompt);
        var uncondTokens1 = _tokenizer1.CreateUnconditionedInput();
        var uncondTokens2 = _tokenizer2.CreateUnconditionedInput();

        // 2. Encode text
        Console.WriteLine("Encoding text...");
        var (hiddenState1, _) = _textEncoder1.Encode(tokens1);
        var (hiddenState2, pooledOutput2) = _textEncoder2.Encode(tokens2);
        var (uncondHidden1, _) = _textEncoder1.Encode(uncondTokens1);
        var (uncondHidden2, uncondPooled2) = _textEncoder2.Encode(uncondTokens2);

        // Concatenate hidden states along last axis: 768 + 1280 = 2048
        var promptEmbeds = TensorHelper.Concatenate(hiddenState1, hiddenState2, axis: 2);
        var negativeEmbeds = TensorHelper.Concatenate(uncondHidden1, uncondHidden2, axis: 2);

        // For classifier-free guidance: concat [negative, positive] on batch dim
        var encoderHiddenStates = TensorHelper.Concatenate(negativeEmbeds, promptEmbeds, axis: 0);

        // Pooled output from encoder 2
        var promptPooled = pooledOutput2
            ?? throw new InvalidOperationException("Text encoder 2 did not return pooled output");
        var negativePooled = uncondPooled2
            ?? throw new InvalidOperationException("Text encoder 2 did not return pooled output for unconditioned input");
        var textEmbeds = TensorHelper.Concatenate(negativePooled, promptPooled, axis: 0);

        // 3. Prepare micro-conditioning time_ids: [original_h, original_w, crop_top, crop_left, target_h, target_w]
        var timeIdValues = new float[]
        {
            _config.Height, _config.Width, 0f, 0f, _config.Height, _config.Width,
            _config.Height, _config.Width, 0f, 0f, _config.Height, _config.Width
        };
        var timeIds = new DenseTensor<float>(timeIdValues, [2, 6]);

        // 4. Setup scheduler
        _scheduler.SetTimesteps(_config.NumInferenceSteps);

        // 5. Generate initial latent noise
        int latentHeight = _config.Height / 8;
        int latentWidth = _config.Width / 8;
        var latents = TensorHelper.CreateRandomLatent(_config.Seed, 1, 4, latentHeight, latentWidth);

        // Scale initial noise by InitNoiseSigma
        latents = TensorHelper.ScalarMultiply(latents, _scheduler.InitNoiseSigma);

        // 6. Denoising loop
        Console.WriteLine($"Running {_config.NumInferenceSteps} denoising steps...");
        var stepSw = Stopwatch.StartNew();

        for (int i = 0; i < _config.NumInferenceSteps; i++)
        {
            stepSw.Restart();

            // Duplicate latent for classifier-free guidance (negative + positive)
            var latentInput = TensorHelper.Duplicate(latents);

            // Scale model input
            latentInput = _scheduler.ScaleModelInput(latentInput, i);

            // Run U-Net
            float timestep = _scheduler.Timesteps[i];
            var noisePred = _unet.Predict(latentInput, timestep, encoderHiddenStates, textEmbeds, timeIds);

            // Apply classifier-free guidance
            var noisePredUncond = TensorHelper.SliceBatch(noisePred, 0);
            var noisePredText = TensorHelper.SliceBatch(noisePred, 1);
            var guidanceDiff = TensorHelper.Subtract(noisePredText, noisePredUncond);
            var guidanceScaled = TensorHelper.ScalarMultiply(guidanceDiff, _config.GuidanceScale);
            var guidedNoisePred = TensorHelper.Add(noisePredUncond, guidanceScaled);

            // Scheduler step
            latents = _scheduler.Step(guidedNoisePred, i, latents);

            Console.WriteLine($"  Step {i + 1}/{_config.NumInferenceSteps} ({stepSw.ElapsedMilliseconds}ms)");
        }

        // 7. Decode latents with VAE
        Console.WriteLine("Decoding latents with VAE...");
        var decodeSw = Stopwatch.StartNew();
        var image = _vaeDecoder.Decode(latents);
        Console.WriteLine($"  VAE decode: {decodeSw.ElapsedMilliseconds}ms");

        // 8. Save image
        Console.WriteLine($"Saving image to {_config.OutputPath}...");
        ImageHelper.SaveTensorAsImage(image, _config.OutputPath);

        Console.WriteLine($"Done! Total time: {totalSw.Elapsed.TotalSeconds:F1}s");
    }

    private static SessionOptions CreateSessionOptions()
    {
        var options = new SessionOptions();

        try
        {
            options.AppendExecutionProvider_DML(0);
            Console.WriteLine("Using DirectML execution provider");
        }
        catch
        {
            Console.WriteLine("DirectML not available, falling back to CPU");
        }

        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

        return options;
    }

    public void Dispose()
    {
        _textEncoder1.Dispose();
        _textEncoder2.Dispose();
        _unet.Dispose();
        _vaeDecoder.Dispose();
    }
}
