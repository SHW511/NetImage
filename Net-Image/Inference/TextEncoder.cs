using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Net_Image.Inference;

public sealed class TextEncoder : IDisposable
{
    private readonly InferenceSession _session;
    private readonly int _hiddenSize;

    public TextEncoder(string modelPath, SessionOptions sessionOptions, int hiddenSize)
    {
        _session = new InferenceSession(modelPath, sessionOptions);
        _hiddenSize = hiddenSize;
    }

    /// <summary>
    /// Runs the text encoder and returns (lastHiddenState, pooledOutput).
    /// pooledOutput is only meaningful for encoder 2 (SDXL).
    /// </summary>
    public (DenseTensor<float> HiddenState, DenseTensor<float>? PooledOutput) Encode(long[] tokenIds)
    {
        var inputTensor = new DenseTensor<long>(tokenIds, [1, tokenIds.Length]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputTensor)
        };

        using var results = _session.Run(inputs);

        var hiddenState = CopyTensor(results.First(r => r.Name == "last_hidden_state"));

        DenseTensor<float>? pooledOutput = null;
        var pooledResult = results.FirstOrDefault(r => r.Name is "text_embeds" or "pooler_output");
        if (pooledResult is not null)
            pooledOutput = CopyTensor(pooledResult);

        return (hiddenState, pooledOutput);
    }

    private static DenseTensor<float> CopyTensor(DisposableNamedOnnxValue value)
    {
        var source = value.AsEnumerable<float>().ToArray();
        var dims = value.AsTensor<float>().Dimensions.ToArray();
        var tensor = new DenseTensor<float>(dims);
        source.AsSpan().CopyTo(tensor.Buffer.Span);
        return tensor;
    }

    public void Dispose() => _session.Dispose();
}
