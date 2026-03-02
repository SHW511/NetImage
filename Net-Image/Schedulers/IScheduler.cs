using Microsoft.ML.OnnxRuntime.Tensors;

namespace Net_Image.Schedulers;

public interface IScheduler
{
    float InitNoiseSigma { get; }
    float[] Timesteps { get; }

    void SetTimesteps(int numInferenceSteps);

    DenseTensor<float> ScaleModelInput(DenseTensor<float> sample, int stepIndex);

    DenseTensor<float> Step(DenseTensor<float> modelOutput, int stepIndex, DenseTensor<float> sample);
}
