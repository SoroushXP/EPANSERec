using EPANSERec.Core.Recommendation;
using EPANSERec.Core.Utils;

Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║  EPAN-SERec: Expertise Preference-Aware Networks for         ║");
Console.WriteLine("║  Software Expert Recommendations with Knowledge Graph        ║");
Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
Console.WriteLine();

// Configuration - Tuned hyperparameters
var config = new TrainingConfig
{
    Epochs = 100,           // Increased for better convergence
    BatchSize = 32,         // Smaller batch for better gradient estimates
    LearningRate = 5e-4f,   // Slightly higher learning rate
    TransHEpochs = 100,     // More pre-training for better embeddings
    EPDRLEpisodes = 50,     // More episodes for better preference graphs
    GCNEpochs = 50,         // More GCN epochs
    EarlyStoppingPatience = 15, // More patience
    Seed = 42
};

Console.WriteLine("Configuration:");
Console.WriteLine($"  - Epochs: {config.Epochs}");
Console.WriteLine($"  - Batch Size: {config.BatchSize}");
Console.WriteLine($"  - Learning Rate: {config.LearningRate}");
Console.WriteLine($"  - TransH Epochs: {config.TransHEpochs}");
Console.WriteLine($"  - EPDRL Episodes: {config.EPDRLEpisodes}");
Console.WriteLine($"  - GCN Epochs: {config.GCNEpochs}");
Console.WriteLine();

// Generate sample data - larger and more balanced dataset
Console.WriteLine("Generating sample StackOverflow-like dataset...");
var dataGenerator = new SampleDataGenerator(config.Seed);
var (knowledgeGraph, experts, questions, samples) = dataGenerator.GenerateDataset(
    numExperts: 100,        // More experts
    numQuestions: 500,      // More questions
    numSamples: 3000        // More training samples
);

Console.WriteLine($"  - Knowledge Graph: {knowledgeGraph.Entities.Count} entities, {knowledgeGraph.Triples.Count} triples");
Console.WriteLine($"  - Experts: {experts.Count}");
Console.WriteLine($"  - Questions: {questions.Count}");
Console.WriteLine($"  - Training Samples: {samples.Count}");
Console.WriteLine();

// Create and train model
var model = new EPANSERecModel(knowledgeGraph, embeddingDim: 64, beta: 0.1f);
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
