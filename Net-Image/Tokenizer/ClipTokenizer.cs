using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Net_Image.Tokenizer;

public partial class ClipTokenizer
{
    private const int MaxLength = 77;
    private const string StartToken = "<|startoftext|>";
    private const string EndToken = "<|endoftext|>";

    private readonly Dictionary<string, int> _encoder;
    private readonly List<(string, string)> _bpeMerges;
    private readonly Dictionary<(string, string), int> _bpeRanks;
    private readonly int _startTokenId;
    private readonly int _endTokenId;
    private readonly Dictionary<string, string[]> _bpeCache = new();

    public ClipTokenizer(string vocabPath, string mergesPath)
    {
        // Load vocab.json
        var vocabJson = File.ReadAllText(vocabPath);
        _encoder = JsonSerializer.Deserialize<Dictionary<string, int>>(vocabJson)
            ?? throw new InvalidOperationException("Failed to parse vocab.json");

        _startTokenId = _encoder[StartToken];
        _endTokenId = _encoder[EndToken];

        // Load merges.txt — skip the header line
        var mergeLines = File.ReadAllLines(mergesPath)
            .Where(line => !string.IsNullOrWhiteSpace(line) && !line.StartsWith("#version"))
            .ToList();

        _bpeMerges = [];
        _bpeRanks = [];
        for (int i = 0; i < mergeLines.Count; i++)
        {
            var parts = mergeLines[i].Split(' ', 2);
            if (parts.Length == 2)
            {
                var pair = (parts[0], parts[1]);
                _bpeMerges.Add(pair);
                _bpeRanks[pair] = i;
            }
        }
    }

    public long[] Tokenize(string text)
    {
        var tokens = new List<int> { _startTokenId };

        // CLIP-style preprocessing: lowercase and clean
        text = ClipPattern().Replace(text.ToLowerInvariant().Trim(), " ");

        // Tokenize using CLIP's regex pattern
        var matches = TokenPattern().Matches(text);
        foreach (Match match in matches)
        {
            var word = match.Value;
            var bpeTokens = BpeEncode(word);
            foreach (var token in bpeTokens)
            {
                if (_encoder.TryGetValue(token, out int id))
                    tokens.Add(id);
            }
        }

        tokens.Add(_endTokenId);

        // Truncate to MaxLength
        if (tokens.Count > MaxLength)
            tokens = tokens.Take(MaxLength - 1).Append(_endTokenId).ToList();

        // Pad to MaxLength
        while (tokens.Count < MaxLength)
            tokens.Add(_endTokenId);

        return tokens.Select(t => (long)t).ToArray();
    }

    public long[] CreateUnconditionedInput()
    {
        var tokens = new long[MaxLength];
        tokens[0] = _startTokenId;
        tokens[1] = _endTokenId;
        for (int i = 2; i < MaxLength; i++)
            tokens[i] = _endTokenId;
        return tokens;
    }

    private string[] BpeEncode(string token)
    {
        // Add </w> suffix for CLIP tokenizer
        var word = token + "</w>";

        if (_bpeCache.TryGetValue(word, out var cached))
            return cached;

        var chars = word.Select(c => c.ToString()).ToList();

        while (chars.Count > 1)
        {
            // Find the pair with the lowest rank
            (string, string)? bestPair = null;
            int bestRank = int.MaxValue;

            for (int i = 0; i < chars.Count - 1; i++)
            {
                var pair = (chars[i], chars[i + 1]);
                if (_bpeRanks.TryGetValue(pair, out int rank) && rank < bestRank)
                {
                    bestPair = pair;
                    bestRank = rank;
                }
            }

            if (bestPair is null)
                break;

            var (first, second) = bestPair.Value;
            var merged = first + second;
            var newChars = new List<string>();

            int j = 0;
            while (j < chars.Count)
            {
                if (j < chars.Count - 1 && chars[j] == first && chars[j + 1] == second)
                {
                    newChars.Add(merged);
                    j += 2;
                }
                else
                {
                    newChars.Add(chars[j]);
                    j++;
                }
            }

            chars = newChars;
        }

        var result = chars.ToArray();
        _bpeCache[word] = result;
        return result;
    }

    [GeneratedRegex(@"\s+")]
    private static partial Regex ClipPattern();

    [GeneratedRegex(@"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+", RegexOptions.IgnoreCase)]
    private static partial Regex TokenPattern();
}
