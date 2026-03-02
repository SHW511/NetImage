using Microsoft.ML.OnnxRuntime.Tensors;

namespace Net_Image.Schedulers;

public class EulerDiscreteScheduler : IScheduler
{
    private const int TrainTimesteps = 1000;
    private const float BetaStart = 0.00085f;
    private const float BetaEnd = 0.012f;

    private float[] _sigmas = [];
    private float[] _timesteps = [];

    public float InitNoiseSigma { get; private set; }
    public float[] Timesteps => _timesteps;

    public void SetTimesteps(int numInferenceSteps)
    {
        // Compute betas using scaled linear schedule
        var betas = new float[TrainTimesteps];
        float sqrtStart = MathF.Sqrt(BetaStart);
        float sqrtEnd = MathF.Sqrt(BetaEnd);
        for (int i = 0; i < TrainTimesteps; i++)
        {
            float t = (float)i / (TrainTimesteps - 1);
            float sqrtBeta = sqrtStart + t * (sqrtEnd - sqrtStart);
            betas[i] = sqrtBeta * sqrtBeta;
        }

        // Compute alphas_cumprod
        var alphasCumprod = new float[TrainTimesteps];
        float cumprod = 1f;
        for (int i = 0; i < TrainTimesteps; i++)
        {
            cumprod *= (1f - betas[i]);
            alphasCumprod[i] = cumprod;
        }

        // Compute sigmas from alphas_cumprod
        var allSigmas = new float[TrainTimesteps];
        for (int i = 0; i < TrainTimesteps; i++)
            allSigmas[i] = MathF.Sqrt((1f - alphasCumprod[i]) / alphasCumprod[i]);

        // Interpolate timesteps
        var stepRatio = (float)(TrainTimesteps - 1) / (numInferenceSteps - 1);
        _timesteps = new float[numInferenceSteps];
        _sigmas = new float[numInferenceSteps + 1];

        for (int i = 0; i < numInferenceSteps; i++)
        {
            float idx = i * stepRatio;
            int low = (int)idx;
            int high = Math.Min(low + 1, TrainTimesteps - 1);
            float frac = idx - low;

            _timesteps[numInferenceSteps - 1 - i] = idx;
            float sigma = allSigmas[low] * (1f - frac) + allSigmas[high] * frac;
            _sigmas[numInferenceSteps - 1 - i] = sigma;
        }
        _sigmas[numInferenceSteps] = 0f; // terminal sigma

        InitNoiseSigma = _sigmas[0];
    }

    public DenseTensor<float> ScaleModelInput(DenseTensor<float> sample, int stepIndex)
    {
        float sigma = _sigmas[stepIndex];
        float scale = 1f / MathF.Sqrt(sigma * sigma + 1f);

        var result = new DenseTensor<float>(sample.Dimensions.ToArray());
        var src = sample.Buffer.Span;
        var dst = result.Buffer.Span;
        for (int i = 0; i < src.Length; i++)
            dst[i] = src[i] * scale;

        return result;
    }

    public DenseTensor<float> Step(DenseTensor<float> modelOutput, int stepIndex, DenseTensor<float> sample)
    {
        float sigma = _sigmas[stepIndex];
        float sigmaNext = _sigmas[stepIndex + 1];

        // Convert model output to "original sample" prediction (v-prediction or epsilon)
        // For Euler discrete with epsilon prediction:
        // pred_original = sample - sigma * model_output
        // derivative = (sample - pred_original) / sigma = model_output
        // dt = sigma_next - sigma
        // prev_sample = sample + derivative * dt

        float dt = sigmaNext - sigma;

        var result = new DenseTensor<float>(sample.Dimensions.ToArray());
        var sampleSpan = sample.Buffer.Span;
        var outputSpan = modelOutput.Buffer.Span;
        var dst = result.Buffer.Span;

        for (int i = 0; i < sampleSpan.Length; i++)
            dst[i] = sampleSpan[i] + outputSpan[i] * dt;

        return result;
    }
}
