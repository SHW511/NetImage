using Microsoft.ML.OnnxRuntime.Tensors;

namespace Net_Image.Utils;

public static class TensorHelper
{
    public static DenseTensor<float> CreateRandomLatent(int seed, int batchSize, int channels, int height, int width)
    {
        var random = new Random(seed);
        var tensor = new DenseTensor<float>([batchSize, channels, height, width]);

        for (int i = 0; i < tensor.Length; i++)
        {
            // Box-Muller transform for Gaussian noise
            double u1 = 1.0 - random.NextDouble();
            double u2 = random.NextDouble();
            tensor.Buffer.Span[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        return tensor;
    }

    public static DenseTensor<float> Duplicate(DenseTensor<float> tensor)
    {
        var dims = tensor.Dimensions.ToArray();
        dims[0] *= 2;
        var result = new DenseTensor<float>(dims);
        var src = tensor.Buffer.Span;
        var dst = result.Buffer.Span;
        int singleLen = (int)tensor.Length;

        src.CopyTo(dst);
        src.CopyTo(dst[singleLen..]);

        return result;
    }

    public static DenseTensor<float> ScalarMultiply(DenseTensor<float> tensor, float scalar)
    {
        var result = new DenseTensor<float>(tensor.Dimensions.ToArray());
        var src = tensor.Buffer.Span;
        var dst = result.Buffer.Span;

        for (int i = 0; i < src.Length; i++)
            dst[i] = src[i] * scalar;

        return result;
    }

    public static DenseTensor<float> Add(DenseTensor<float> a, DenseTensor<float> b)
    {
        var result = new DenseTensor<float>(a.Dimensions.ToArray());
        var spanA = a.Buffer.Span;
        var spanB = b.Buffer.Span;
        var dst = result.Buffer.Span;

        for (int i = 0; i < spanA.Length; i++)
            dst[i] = spanA[i] + spanB[i];

        return result;
    }

    public static DenseTensor<float> Subtract(DenseTensor<float> a, DenseTensor<float> b)
    {
        var result = new DenseTensor<float>(a.Dimensions.ToArray());
        var spanA = a.Buffer.Span;
        var spanB = b.Buffer.Span;
        var dst = result.Buffer.Span;

        for (int i = 0; i < spanA.Length; i++)
            dst[i] = spanA[i] - spanB[i];

        return result;
    }

    public static DenseTensor<float> Concatenate(DenseTensor<float> a, DenseTensor<float> b, int axis)
    {
        var dimsA = a.Dimensions.ToArray();
        var dimsB = b.Dimensions.ToArray();
        var dimsOut = (int[])dimsA.Clone();
        dimsOut[axis] = dimsA[axis] + dimsB[axis];

        var result = new DenseTensor<float>(dimsOut);

        if (axis == dimsA.Length - 1)
        {
            // Fast path for last-axis concatenation (common for hidden state concat)
            int outerSize = 1;
            for (int d = 0; d < axis; d++)
                outerSize *= dimsA[d];

            int innerA = dimsA[axis];
            int innerB = dimsB[axis];
            int innerOut = innerA + innerB;

            var srcA = a.Buffer.Span;
            var srcB = b.Buffer.Span;
            var dst = result.Buffer.Span;

            for (int outer = 0; outer < outerSize; outer++)
            {
                srcA.Slice(outer * innerA, innerA).CopyTo(dst.Slice(outer * innerOut, innerA));
                srcB.Slice(outer * innerB, innerB).CopyTo(dst.Slice(outer * innerOut + innerA, innerB));
            }
        }
        else if (axis == 0)
        {
            var srcA = a.Buffer.Span;
            var srcB = b.Buffer.Span;
            var dst = result.Buffer.Span;
            srcA.CopyTo(dst);
            srcB.CopyTo(dst[(int)a.Length..]);
        }
        else
        {
            throw new NotSupportedException($"Concatenation on axis {axis} not implemented for rank {dimsA.Length}");
        }

        return result;
    }

    public static DenseTensor<float> SliceBatch(DenseTensor<float> tensor, int batchIndex)
    {
        var dims = tensor.Dimensions.ToArray();
        int singleLen = 1;
        for (int d = 1; d < dims.Length; d++)
            singleLen *= dims[d];

        var outDims = (int[])dims.Clone();
        outDims[0] = 1;

        var result = new DenseTensor<float>(outDims);
        tensor.Buffer.Span.Slice(batchIndex * singleLen, singleLen).CopyTo(result.Buffer.Span);

        return result;
    }
}
