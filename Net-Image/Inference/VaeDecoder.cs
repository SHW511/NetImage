using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Net_Image.Inference;

public sealed class VaeDecoder : IDisposable
{
    private const float ScaleFactor = 0.13025f;

    private readonly InferenceSession _session;

    public VaeDecoder(string modelPath, SessionOptions sessionOptions)
    {
        _session = new InferenceSession(modelPath, sessionOptions);
    }

    /// <summary>
    /// Decodes latent representation to image tensor.
    /// Applies the VAE scaling factor before decoding.
    /// </summary>
    public DenseTensor<float> Decode(DenseTensor<float> latent)
    {
        // Scale the latent by 1/scale_factor
        var scaled = new DenseTensor<float>(latent.Dimensions.ToArray());
        var src = latent.Buffer.Span;
        var dst = scaled.Buffer.Span;
        float invScale = 1f / ScaleFactor;
        for (int i = 0; i < src.Length; i++)
            dst[i] = src[i] * invScale;

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("latent_sample", scaled)
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
