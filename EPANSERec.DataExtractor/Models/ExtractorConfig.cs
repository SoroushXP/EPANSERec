using Microsoft.Extensions.Configuration;

namespace EPANSERec.DataExtractor.Models;

/// <summary>
/// Configuration for the StackOverflow data extractor.
/// </summary>
public class ExtractorConfig
{
    /// <summary>
    /// Input directory path containing StackOverflow data dump XML files.
    /// </summary>
    public string InputPath { get; set; } = "";

    /// <summary>
    /// Path to the Posts.xml file from StackOverflow data dump.
    /// </summary>
    public string PostsXmlPath { get; set; } = "";

    /// <summary>
    /// Path to the Users.xml file from StackOverflow data dump.
    /// </summary>
    public string UsersXmlPath { get; set; } = "";

    /// <summary>
    /// Path to the Tags.xml file from StackOverflow data dump.
    /// </summary>
    public string TagsXmlPath { get; set; } = "";

    /// <summary>
    /// Output directory for extracted data files.
    /// </summary>
    public string OutputPath { get; set; } = "data/stackoverflow";

    /// <summary>
    /// Maximum number of questions to extract (-1 for unlimited).
    /// </summary>
    public int MaxQuestions { get; set; } = 10000;

    /// <summary>
    /// Maximum number of experts to include (-1 for unlimited).
    /// </summary>
    public int MaxExperts { get; set; } = 1000;

    /// <summary>
    /// Minimum number of answers required to be considered an expert.
    /// </summary>
    public int MinExpertAnswers { get; set; } = 5;

    /// <summary>
    /// Minimum tag occurrences to include a tag as an entity.
    /// </summary>
    public int MinTagOccurrences { get; set; } = 100;

    /// <summary>
    /// List of software-related tags to filter for (loaded from config).
    /// </summary>
    public List<string> SoftwareTags { get; set; } = new();

    /// <summary>
    /// Gets the software tags as a HashSet for efficient lookup.
    /// </summary>
    public HashSet<string> GetSoftwareTagsSet()
    {
        if (SoftwareTags.Count == 0)
            return GetDefaultSoftwareTags();
        return new HashSet<string>(SoftwareTags, StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Loads configuration from appsettings.json file.
    /// </summary>
    public static ExtractorConfig LoadFromFile(string? configPath = null)
    {
        var basePath = configPath ?? AppContext.BaseDirectory;
        var configFile = Path.Combine(basePath, "appsettings.json");

        if (!File.Exists(configFile))
        {
            Console.WriteLine($"  Config file not found at {configFile}, using defaults");
            return new ExtractorConfig();
        }

        var configuration = new ConfigurationBuilder()
            .SetBasePath(basePath)
            .AddJsonFile("appsettings.json", optional: true, reloadOnChange: false)
            .Build();

        var config = new ExtractorConfig();
        configuration.GetSection("Extractor").Bind(config);

        // Set derived paths if InputPath is provided
        if (!string.IsNullOrEmpty(config.InputPath))
        {
            config.PostsXmlPath = Path.Combine(config.InputPath, "Posts.xml");
            config.UsersXmlPath = Path.Combine(config.InputPath, "Users.xml");
            config.TagsXmlPath = Path.Combine(config.InputPath, "Tags.xml");
        }

        return config;
    }

    /// <summary>
    /// Creates configuration with the specified values, falling back to defaults.
    /// </summary>
    public static ExtractorConfig CreateDefault(string inputPath, string outputPath,
        int maxQuestions = 10000, int maxExperts = 1000, int minAnswers = 5)
    {
        return new ExtractorConfig
        {
            InputPath = inputPath,
            PostsXmlPath = Path.Combine(inputPath, "Posts.xml"),
            UsersXmlPath = Path.Combine(inputPath, "Users.xml"),
            TagsXmlPath = Path.Combine(inputPath, "Tags.xml"),
            OutputPath = outputPath,
            MaxQuestions = maxQuestions,
            MaxExperts = maxExperts,
            MinExpertAnswers = minAnswers,
            MinTagOccurrences = 100,
            SoftwareTags = GetDefaultSoftwareTags().ToList()
        };
    }

    /// <summary>
    /// Applies command-line overrides to the configuration.
    /// </summary>
    public void ApplyOverrides(string? inputPath, string? outputPath, int? maxQuestions, int? maxExperts, int? minAnswers)
    {
        if (!string.IsNullOrEmpty(inputPath))
        {
            InputPath = inputPath;
            PostsXmlPath = Path.Combine(inputPath, "Posts.xml");
            UsersXmlPath = Path.Combine(inputPath, "Users.xml");
            TagsXmlPath = Path.Combine(inputPath, "Tags.xml");
        }

        if (!string.IsNullOrEmpty(outputPath))
            OutputPath = outputPath;

        if (maxQuestions.HasValue)
            MaxQuestions = maxQuestions.Value;

        if (maxExperts.HasValue)
            MaxExperts = maxExperts.Value;

        if (minAnswers.HasValue)
            MinExpertAnswers = minAnswers.Value;
    }

    /// <summary>
    /// Returns the default set of software-related tags.
    /// </summary>
    public static HashSet<string> GetDefaultSoftwareTags()
    {
        return new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            // Programming Languages
            "python", "javascript", "java", "c#", "c++", "php", "ruby", "go", "rust", "swift",
            "kotlin", "typescript", "scala", "r", "matlab", "perl", "haskell", "lua",

            // Frameworks
            "django", "flask", "react", "angular", "vue.js", "node.js", "express", "spring",
            "asp.net", "asp.net-mvc", "asp.net-core", ".net", "rails", "laravel", "symfony",

            // Databases
            "sql", "mysql", "postgresql", "mongodb", "sqlite", "redis", "elasticsearch",
            "sql-server", "oracle", "cassandra", "firebase",

            // Tools & Technologies
            "git", "docker", "kubernetes", "aws", "azure", "linux", "nginx", "apache",
            "rest", "api", "graphql", "json", "xml", "html", "css", "ajax",

            // Concepts
            "algorithm", "data-structures", "machine-learning", "deep-learning", "oop",
            "design-patterns", "unit-testing", "security", "authentication", "oauth"
        };
    }
}

