namespace EPANSERec.DataExtractor;

/// <summary>
/// Provides help text and usage instructions for the data extractor.
/// </summary>
public static class HelpText
{
    /// <summary>
    /// Prints command-line usage help.
    /// </summary>
    public static void PrintUsage()
    {
        Console.WriteLine("Usage: EPANSERec.DataExtractor [options]");
        Console.WriteLine();
        Console.WriteLine("Configuration is loaded from appsettings.json. Command-line options override config file.");
        Console.WriteLine();
        Console.WriteLine("Options:");
        Console.WriteLine("  -i, --input <path>       Path to directory containing Posts.xml, Tags.xml");
        Console.WriteLine("  -o, --output <path>      Output directory (default from config: data/stackoverflow)");
        Console.WriteLine("  -q, --max-questions <n>  Max questions to extract (default from config: 10000)");
        Console.WriteLine("  -e, --max-experts <n>    Max experts to include (default from config: 1000)");
        Console.WriteLine("  -a, --min-answers <n>    Min answers to be an expert (default from config: 5)");
        Console.WriteLine("  -h, --help               Show this help message");
        Console.WriteLine();
        Console.WriteLine("Configuration file (appsettings.json):");
        Console.WriteLine("  Edit appsettings.json to change defaults without recompiling.");
        Console.WriteLine("  Settings include: InputPath, OutputPath, MaxQuestions, MaxExperts,");
        Console.WriteLine("                    MinExpertAnswers, MinTagOccurrences, SoftwareTags[]");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  EPANSERec.DataExtractor -i C:\\data\\stackoverflow");
        Console.WriteLine("  EPANSERec.DataExtractor -i ./stackoverflow -o ./output -q 50000");
        Console.WriteLine();
        Console.WriteLine("If no input path is provided (via CLI or appsettings.json), enters interactive mode.");
        Console.WriteLine();
        PrintDownloadInstructions();
    }

    /// <summary>
    /// Prints instructions for downloading StackOverflow data dumps.
    /// </summary>
    public static void PrintDownloadInstructions()
    {
        Console.WriteLine("To download StackOverflow data:");
        Console.WriteLine("1. Go to: https://archive.org/details/stackexchange");
        Console.WriteLine("2. Download 'stackoverflow.com-Posts.7z' (or a smaller site-specific dump)");
        Console.WriteLine("3. Extract the .7z archive using 7-Zip");
        Console.WriteLine("4. Run this tool with the path to the extracted directory");
        Console.WriteLine();
        Console.WriteLine("Recommended smaller datasets for testing:");
        Console.WriteLine("  - softwareengineering.stackexchange.com (~200MB)");
        Console.WriteLine("  - codereview.stackexchange.com (~150MB)");
        Console.WriteLine("  - programmers.stackexchange.com (~100MB)");
    }
}

