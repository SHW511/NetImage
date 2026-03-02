using Net_Image.Pipeline;

internal class Program
{
    private static void Main(string[] args)
    {
        if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
        {
            PrintUsage();
            return;
        }

        var config = ParseArgs(args);
        using var pipeline = new StableDiffusionPipeline(config);
        pipeline.Generate();

        static PipelineConfig ParseArgs(string[] args)
        {
            string? model = null;
            string? prompt = null;
            string negativePrompt = "";
            string output = "output.png";
            int steps = 30;
            float guidanceScale = 7.5f;
            int seed = 42;
            int width = 1024;
            int height = 1024;

            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--model" or "-m":
                        model = args[++i];
                        break;
                    case "--prompt" or "-p":
                        prompt = args[++i];
                        break;
                    case "--negative-prompt" or "-n":
                        negativePrompt = args[++i];
                        break;
                    case "--output" or "-o":
                        output = args[++i];
                        break;
                    case "--steps" or "-s":
                        steps = int.Parse(args[++i]);
                        break;
                    case "--guidance-scale" or "-g":
                        guidanceScale = float.Parse(args[++i]);
                        break;
                    case "--seed":
                        seed = int.Parse(args[++i]);
                        break;
                    case "--width":
                        width = int.Parse(args[++i]);
                        break;
                    case "--height":
                        height = int.Parse(args[++i]);
                        break;
                }
            }

            if (model is null)
            {
                Console.Error.WriteLine("Error: --model is required");
                PrintUsage();
                Environment.Exit(1);
            }

            if (prompt is null)
            {
                Console.Error.WriteLine("Error: --prompt is required");
                PrintUsage();
                Environment.Exit(1);
            }

            return new PipelineConfig
            {
                ModelDirectory = model,
                Prompt = prompt,
                NegativePrompt = negativePrompt,
                OutputPath = output,
                NumInferenceSteps = steps,
                GuidanceScale = guidanceScale,
                Seed = seed,
                Width = width,
                Height = height,
            };
        }

        static void PrintUsage()
        {
            Console.WriteLine("""
        SDXL Text-to-Image Generator (ONNX Runtime + CUDA)

        Usage:
          Net-Image --model <path> --prompt <text> [options]

        Required:
          --model, -m <path>         Path to SDXL ONNX model directory
          --prompt, -p <text>        Text prompt for image generation

        Options:
          --negative-prompt, -n <text>  Negative prompt (default: "")
          --output, -o <path>           Output PNG file path (default: output.png)
          --steps, -s <int>             Number of inference steps (default: 30)
          --guidance-scale, -g <float>  Classifier-free guidance scale (default: 7.5)
          --seed <int>                  Random seed (default: 42)
          --width <int>                 Output width in pixels (default: 1024)
          --height <int>                Output height in pixels (default: 1024)
          --help, -h                    Show this help message
        """);
        }
    }
}