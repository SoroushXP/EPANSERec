using EPANSERec.DataExtractor;
using EPANSERec.DataExtractor.Models;

Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║  StackOverflow Data Extractor for EPAN-SERec                 ║");
Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
Console.WriteLine();

// Load configuration from appsettings.json
Console.WriteLine("Loading configuration from appsettings.json...");
var config = ExtractorConfig.LoadFromFile();

// Parse command-line arguments (override config file settings)
string? inputPath = null;
string? outputPath = null;
int? maxQuestions = null;
int? maxExperts = null;
int? minAnswers = null;

for (int i = 0; i < args.Length; i++)
{
    switch (args[i].ToLower())
    {
        case "-i":
        case "--input":
            if (i + 1 < args.Length) inputPath = args[++i];
            break;
        case "-o":
        case "--output":
            if (i + 1 < args.Length) outputPath = args[++i];
            break;
        case "-q":
        case "--max-questions":
            if (i + 1 < args.Length && int.TryParse(args[i + 1], out var q))
            {
                maxQuestions = q;
                i++;
            }
            break;
        case "-e":
        case "--max-experts":
            if (i + 1 < args.Length && int.TryParse(args[i + 1], out var e))
            {
                maxExperts = e;
                i++;
            }
            break;
        case "-a":
        case "--min-answers":
            if (i + 1 < args.Length && int.TryParse(args[i + 1], out var a))
            {
                minAnswers = a;
                i++;
            }
            break;
        case "-h":
        case "--help":
            HelpText.PrintUsage();
            return;
    }
}

// Apply command-line overrides to config
config.ApplyOverrides(inputPath, outputPath, maxQuestions, maxExperts, minAnswers);

// If no input path provided (neither from config nor CLI), enter interactive mode
if (string.IsNullOrEmpty(config.InputPath))
{
    await InteractiveMode.RunAsync(config);
    return;
}

Console.WriteLine();
Console.WriteLine("Configuration:");
Console.WriteLine($"  Input Directory: {config.InputPath}");
Console.WriteLine($"  Posts XML: {config.PostsXmlPath}");
Console.WriteLine($"  Tags XML: {config.TagsXmlPath}");
Console.WriteLine($"  Output: {config.OutputPath}");
Console.WriteLine($"  Max Questions: {config.MaxQuestions}");
Console.WriteLine($"  Max Experts: {config.MaxExperts}");
Console.WriteLine($"  Min Expert Answers: {config.MinExpertAnswers}");
Console.WriteLine($"  Software Tags: {config.GetSoftwareTagsSet().Count} configured");
Console.WriteLine();

// Check if input files exist
if (!File.Exists(config.PostsXmlPath))
{
    Console.WriteLine($"ERROR: Posts.xml not found at: {config.PostsXmlPath}");
    Console.WriteLine();
    Console.WriteLine("Make sure you have extracted the StackOverflow data dump.");
    Console.WriteLine("The directory should contain Posts.xml (and optionally Tags.xml, Users.xml)");
    Console.WriteLine();
    HelpText.PrintDownloadInstructions();
    return;
}

var extractor = new StackOverflowExtractor(config);
await extractor.ExtractAsync();

Console.WriteLine();
Console.WriteLine("Extraction complete! Data saved to: " + config.OutputPath);
Console.WriteLine();
Console.WriteLine("To use this data, set UseRealData = true in EPANSERec.Training/Program.cs");