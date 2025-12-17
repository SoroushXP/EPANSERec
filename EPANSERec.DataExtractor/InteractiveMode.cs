using EPANSERec.DataExtractor.Models;

namespace EPANSERec.DataExtractor;

/// <summary>
/// Provides interactive command-line interface for the data extractor.
/// </summary>
public static class InteractiveMode
{
    /// <summary>
    /// Runs the interactive command-line mode with loaded configuration.
    /// </summary>
    public static async Task RunAsync(ExtractorConfig config)
    {
        Console.WriteLine("Interactive Mode - Type 'help' for commands, 'exit' to quit");
        Console.WriteLine();
        Console.WriteLine("Configuration loaded from appsettings.json. Use commands to override.");

        while (true)
        {
            Console.WriteLine();
            Console.Write("DataExtractor> ");
            var input = Console.ReadLine()?.Trim();

            if (string.IsNullOrEmpty(input)) continue;

            var parts = input.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            var command = parts[0].ToLower();

            switch (command)
            {
                case "exit":
                case "quit":
                case "q":
                    Console.WriteLine("Goodbye!");
                    return;

                case "help":
                case "h":
                case "?":
                    PrintHelp();
                    break;

                case "extract":
                case "e":
                    if (parts.Length < 2)
                    {
                        Console.WriteLine("Usage: extract <path-to-stackoverflow-data>");
                        break;
                    }
                    var path = string.Join(" ", parts.Skip(1));
                    config.ApplyOverrides(path, null, null, null, null);
                    await ExtractDataAsync(config);
                    break;

                case "output":
                case "o":
                    if (parts.Length < 2)
                        Console.WriteLine($"Current output path: {config.OutputPath}");
                    else
                    {
                        config.OutputPath = parts[1];
                        Console.WriteLine($"Output path set to: {config.OutputPath}");
                    }
                    break;

                case "maxq":
                    if (parts.Length < 2)
                        Console.WriteLine($"Current max questions: {config.MaxQuestions}");
                    else if (int.TryParse(parts[1], out var q))
                    {
                        config.MaxQuestions = q;
                        Console.WriteLine($"Max questions set to: {config.MaxQuestions}");
                    }
                    break;

                case "maxe":
                    if (parts.Length < 2)
                        Console.WriteLine($"Current max experts: {config.MaxExperts}");
                    else if (int.TryParse(parts[1], out var e))
                    {
                        config.MaxExperts = e;
                        Console.WriteLine($"Max experts set to: {config.MaxExperts}");
                    }
                    break;

                case "mina":
                    if (parts.Length < 2)
                        Console.WriteLine($"Current min answers: {config.MinExpertAnswers}");
                    else if (int.TryParse(parts[1], out var a))
                    {
                        config.MinExpertAnswers = a;
                        Console.WriteLine($"Min answers set to: {config.MinExpertAnswers}");
                    }
                    break;

                case "config":
                case "c":
                    Console.WriteLine($"Current Configuration:");
                    Console.WriteLine($"  Input Path: {config.InputPath}");
                    Console.WriteLine($"  Output Path: {config.OutputPath}");
                    Console.WriteLine($"  Max Questions: {config.MaxQuestions}");
                    Console.WriteLine($"  Max Experts: {config.MaxExperts}");
                    Console.WriteLine($"  Min Answers: {config.MinExpertAnswers}");
                    Console.WriteLine($"  Min Tag Occurrences: {config.MinTagOccurrences}");
                    Console.WriteLine($"  Software Tags: {config.GetSoftwareTagsSet().Count} configured");
                    break;

                case "download":
                case "d":
                    HelpText.PrintDownloadInstructions();
                    break;

                case "reload":
                case "r":
                    config = ExtractorConfig.LoadFromFile();
                    Console.WriteLine("Configuration reloaded from appsettings.json");
                    break;

                default:
                    // Treat as a path if it looks like one
                    if (Directory.Exists(input) || input.Contains("\\") || input.Contains("/"))
                    {
                        config.ApplyOverrides(input, null, null, null, null);
                        await ExtractDataAsync(config);
                    }
                    else
                    {
                        Console.WriteLine($"Unknown command: {command}. Type 'help' for available commands.");
                    }
                    break;
            }
        }
    }

    private static async Task ExtractDataAsync(ExtractorConfig config)
    {
        if (!File.Exists(config.PostsXmlPath))
        {
            Console.WriteLine($"ERROR: Posts.xml not found at: {config.PostsXmlPath}");
            Console.WriteLine("Make sure the path contains the extracted StackOverflow data dump.");
            return;
        }

        Console.WriteLine();
        Console.WriteLine($"Extracting from: {config.InputPath}");
        Console.WriteLine($"Output to: {config.OutputPath}");
        Console.WriteLine();

        var extractor = new StackOverflowExtractor(config);
        await extractor.ExtractAsync();

        Console.WriteLine();
        Console.WriteLine("Extraction complete! Data saved to: " + config.OutputPath);
        Console.WriteLine("To use this data, set UseRealData = true in EPANSERec.Training/Program.cs");
    }

    private static void PrintHelp()
    {
        Console.WriteLine();
        Console.WriteLine("Available Commands:");
        Console.WriteLine("  extract <path>    Extract data from StackOverflow dump at <path>");
        Console.WriteLine("  e <path>          (shorthand for extract)");
        Console.WriteLine();
        Console.WriteLine("  config            Show current configuration (from appsettings.json + overrides)");
        Console.WriteLine("  reload            Reload configuration from appsettings.json");
        Console.WriteLine("  output <path>     Set output directory");
        Console.WriteLine("  maxq <n>          Set max questions to extract");
        Console.WriteLine("  maxe <n>          Set max experts to include");
        Console.WriteLine("  mina <n>          Set min answers required to be an expert");
        Console.WriteLine();
        Console.WriteLine("  download          Show download instructions");
        Console.WriteLine("  help              Show this help");
        Console.WriteLine("  exit              Exit the program");
        Console.WriteLine();
        Console.WriteLine("You can also just type a path directly to extract from it.");
    }
}

