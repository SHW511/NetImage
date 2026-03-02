using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Net_Image.Utils;

public static class ImageHelper
{
    /// <summary>
    /// Converts a float tensor with shape (1, 3, H, W) and values in [-1, 1] to a PNG file.
    /// </summary>
    public static void SaveTensorAsImage(DenseTensor<float> tensor, string outputPath)
    {
        int height = tensor.Dimensions[2];
        int width = tensor.Dimensions[3];
        var data = tensor.Buffer.ToArray();
        int channelSize = height * width;

        using var image = new Image<Rgb24>(width, height);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float r = data[0 * channelSize + y * width + x];
                float g = data[1 * channelSize + y * width + x];
                float b = data[2 * channelSize + y * width + x];

                image[x, y] = new Rgb24(
                    ClampToByte((r + 1f) / 2f),
                    ClampToByte((g + 1f) / 2f),
                    ClampToByte((b + 1f) / 2f));
            }
        }

        image.SaveAsPng(outputPath);
    }

    private static byte ClampToByte(float value)
    {
        return (byte)Math.Clamp(value * 255f, 0f, 255f);
    }
}
