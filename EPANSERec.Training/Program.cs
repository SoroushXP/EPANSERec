using EPANSERec.Core.KnowledgeGraph;
using EPANSERec.Core.Models;
using EPANSERec.Core.Recommendation;
using EPANSERec.Core.Utils;

Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║  EPAN-SERec: Expertise Preference-Aware Networks for         ║");
Console.WriteLine("║  Software Expert Recommendations with Knowledge Graph        ║");
Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
Console.WriteLine();

// ============================================================================
// CONFIGURATION - Toggle between synthetic and real StackOverflow data
// ============================================================================
var config = new TrainingConfig
{
    // === Data Source (set UseRealData = true to use real StackOverflow data) ===
    UseRealData = false,                    // Toggle: false = synthetic, true = real data
    DataPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "data", "stackoverflow"),  // Path to real data (when UseRealData = true)

    // === Synthetic Data Settings (only used when UseRealData = false) ===
    NumExperts = 100,
    NumQuestions = 500,
    NumSamples = 3000,

    // === Model Settings ===
    EmbeddingDim = 64,                      // 64 for small data, 100+ for large data
    Beta = 0.1f,                            // SSL loss coefficient
    UseHierarchicalMI = false,              // Enable for large datasets

    // === Training Settings ===
    Epochs = 100,
    BatchSize = 32,
    LearningRate = 5e-4f,
    TransHEpochs = 100,
    EPDRLEpisodes = 50,
    GCNEpochs = 50,
    EarlyStoppingPatience = 0,  // 0 = disabled (train all epochs), >0 = stop if no improvement for N epochs
    Seed = 42
};

// Print configuration
Console.WriteLine("Configuration:");
Console.WriteLine($"  - Data Source: {(config.UseRealData ? "Real StackOverflow" : "Synthetic")}");
if (config.UseRealData)
    Console.WriteLine($"  - Data Path: {config.DataPath}");
Console.WriteLine($"  - Embedding Dim: {config.EmbeddingDim}");
Console.WriteLine($"  - Hierarchical MI: {config.UseHierarchicalMI}");
Console.WriteLine($"  - Epochs: {config.Epochs}");
Console.WriteLine($"  - Batch Size: {config.BatchSize}");
Console.WriteLine($"  - Learning Rate: {config.LearningRate}");
Console.WriteLine();

// ============================================================================
// LOAD DATA - Either synthetic or real StackOverflow data
// ============================================================================
SoftwareKnowledgeGraph knowledgeGraph = null!;
List<Expert> experts = null!;
List<Question> questions = null!;
List<(int questionId, int expertId, bool answered)> samples = null!;
bool dataLoaded = false;

if (config.UseRealData)
{
    // Load real StackOverflow data
    Console.WriteLine("Loading real StackOverflow dataset...");
    var dataLoader = new StackOverflowDataLoader(config.DataPath, config.Seed);

    if (!dataLoader.DatasetExists())
    {
        Console.WriteLine();
        Console.WriteLine("ERROR: StackOverflow dataset not found!");
        Console.WriteLine($"Expected location: {Path.GetFullPath(config.DataPath)}");
        Console.WriteLine();
        Console.WriteLine("Expected directory structure:");
        Console.WriteLine("  {DataPath}/");
        Console.WriteLine("    ├── knowledge_graph/");
        Console.WriteLine("    │   ├── entities.json");
        Console.WriteLine("    │   ├── relations.json");
        Console.WriteLine("    │   └── triples.json");
        Console.WriteLine("    ├── experts.json");
        Console.WriteLine("    ├── questions.json");
        Console.WriteLine("    └── samples.json");
        Console.WriteLine();
        Console.WriteLine("Falling back to synthetic data...");
        Console.WriteLine();
    }
    else
    {
        (knowledgeGraph, experts, questions, samples) = dataLoader.LoadDataset();
        dataLoaded = true;

        // For large datasets, enable hierarchical MI
        if (knowledgeGraph.Entities.Count > 100)
        {
            config.UseHierarchicalMI = true;
            config.EmbeddingDim = 100;  // Larger embeddings for real data
            Console.WriteLine("  Large dataset detected - enabling hierarchical MI");
        }
    }
}

if (!dataLoaded)
{
    // Generate synthetic data
    Console.WriteLine("Generating sample StackOverflow-like dataset...");
    var dataGenerator = new SampleDataGenerator(config.Seed);
    (knowledgeGraph, experts, questions, samples) = dataGenerator.GenerateDataset(
        numExperts: config.NumExperts,
        numQuestions: config.NumQuestions,
        numSamples: config.NumSamples
    );
}

Console.WriteLine($"  - Knowledge Graph: {knowledgeGraph.Entities.Count} entities, {knowledgeGraph.Triples.Count} triples");
Console.WriteLine($"  - Experts: {experts.Count}");
Console.WriteLine($"  - Questions: {questions.Count}");
Console.WriteLine($"  - Training Samples: {samples.Count}");
Console.WriteLine();

// ============================================================================
// CREATE AND TRAIN MODEL
// ============================================================================
var model = new EPANSERecModel(knowledgeGraph, embeddingDim: config.EmbeddingDim, beta: config.Beta);
var pipeline = new TrainingPipeline(model, config);

var result = pipeline.Train(knowledgeGraph, experts, questions, samples);

// Print final results
Console.WriteLine("\n=== Final Results ===");
if (result.ValidationMetrics.Count > 0)
{
    var lastMetrics = result.ValidationMetrics.Last();
    Console.WriteLine($"Final AUC: {lastMetrics.auc:P2}");
    Console.WriteLine($"Final Accuracy: {lastMetrics.acc:P2}");
    Console.WriteLine($"Final F1: {lastMetrics.f1:P2}");
}

// Demo: Recommend experts for a sample question
Console.WriteLine("\n=== Demo: Expert Recommendation ===");
var sampleQuestion = questions[0];
Console.WriteLine($"Question: {sampleQuestion.Title}");
Console.WriteLine($"Tags: {string.Join(", ", sampleQuestion.Tags)}");

var recommendations = model.RecommendExperts(sampleQuestion, experts, topK: 5);
Console.WriteLine("\nTop 5 Recommended Experts:");
for (int i = 0; i < recommendations.Count; i++)
{
    var (expert, score) = recommendations[i];
    Console.WriteLine($"  {i + 1}. {expert.Name} (Score: {score:F4}) - Expertise: {string.Join(", ", expert.ExpertiseTags.Take(3))}");
}

Console.WriteLine("\n=== Training Complete ===");
