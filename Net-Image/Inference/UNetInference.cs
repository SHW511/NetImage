using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Net_Image.Inference;

public sealed class UNetInference : IDisposable
{
    private readonly InferenceSession _session;

    public UNetInference(string modelPath, SessionOptions sessionOptions)
    {
        _session = new InferenceSession(modelPath, sessionOptions);
    }

    /// <summary>
    /// Runs the U-Net denoising step.
    /// </summary>
    /// <param name="sample">Latent sample (batch×4×H×W)</param>
    /// <param name="timestep">Current timestep</param>
    /// <param name="encoderHiddenStates">Text encoder hidden states (batch×77×2048)</param>
    /// <param name="textEmbeds">Pooled text embeddings (batch×1280)</param>
    /// <param name="timeIds">Micro-conditioning time IDs (batch×6)</param>
    public DenseTensor<float> Predict(
        DenseTensor<float> sample,
        float timestep,
        DenseTensor<float> encoderHiddenStates,
        DenseTensor<float> textEmbeds,
        DenseTensor<float> timeIds)
    {
        int batchSize = sample.Dimensions[0];
        var timestepValues = new float[batchSize];
        Array.Fill(timestepValues, timestep);
        var timestepTensor = new DenseTensor<float>(timestepValues, new int[] { batchSize });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("sample", sample),
            NamedOnnxValue.CreateFromTensor("timestep", timestepTensor),
            NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates),
            NamedOnnxValue.CreateFromTensor("text_embeds", textEmbeds),
            NamedOnnxValue.CreateFromTensor("time_ids", timeIds),
        };

        using var results = _session.Run(inputs);
        var output = results.First();
        return CopyOutputTensor(output);
    }

    private static DenseTensor<float> CopyOutputTensor(DisposableNamedOnnxValue value)
    {
        var source = value.AsEnumerable<float>().ToArray();
        var dims = value.AsTensor<float>().Dimensions.ToArray();
        var tensor = new DenseTensor<float>(dims);
        source.AsSpan().CopyTo(tensor.Buffer.Span);
        return tensor;
    }

    public void Dispose() => _session.Dispose();
}
