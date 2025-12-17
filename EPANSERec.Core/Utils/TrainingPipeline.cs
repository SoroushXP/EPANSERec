using EPANSERec.Core.KnowledgeGraph;
using EPANSERec.Core.Models;
using EPANSERec.Core.Recommendation;

namespace EPANSERec.Core.Utils;

/// <summary>
/// Training pipeline for EPAN-SERec model.
/// Orchestrates the complete training process.
/// </summary>
public class TrainingPipeline
{
    private readonly EPANSERecModel _model;
    private readonly TrainingConfig _config;
    private readonly Random _random;

    public TrainingPipeline(EPANSERecModel model, TrainingConfig config)
    {
        _model = model;
        _config = config;
        _random = new Random(config.Seed ?? 42);
    }

    /// <summary>
    /// Runs the complete training pipeline.
    /// </summary>
    public TrainingResult Train(
        SoftwareKnowledgeGraph knowledgeGraph,
        List<Expert> experts,
        List<Question> questions,
        List<(int questionId, int expertId, bool answered)> trainingSamples)
    {
        var result = new TrainingResult();
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        
        Console.WriteLine("=== EPAN-SERec Training Pipeline ===\n");
        
        // Step 1: Initialize model
        Console.WriteLine("Step 1: Initializing model...");
        _model.Initialize(_config.LearningRate);
        
        // Step 2: Pre-train knowledge embeddings
        Console.WriteLine("\nStep 2: Pre-training TransH embeddings...");
        _model.PretrainKnowledgeEmbeddings(_config.TransHEpochs, _config.BatchSize);
        
        // Step 3: Generate expertise preference graphs
        Console.WriteLine("\nStep 3: Generating expertise preference graphs...");
        _model.GenerateExpertPreferenceGraphs(experts, _config.EPDRLEpisodes);
        
        // Step 4: Optimize expert embeddings
        Console.WriteLine("\nStep 4: Optimizing expert embeddings with GCN + SSL...");
        _model.OptimizeExpertEmbeddings(_config.GCNEpochs);
        
        // Step 5: Train prediction model
        Console.WriteLine("\nStep 5: Training prediction model...");
        var expertDict = experts.ToDictionary(e => e.Id);
        var questionDict = questions.ToDictionary(q => q.Id);
        
        // Convert samples to training data
        var trainingData = trainingSamples
            .Where(s => questionDict.ContainsKey(s.questionId) && expertDict.ContainsKey(s.expertId))
            .Select(s => (questionDict[s.questionId], expertDict[s.expertId], s.answered))
            .ToList();
        
        // Split into train/val
        var shuffled = trainingData.OrderBy(_ => _random.Next()).ToList();
        int trainSize = (int)(shuffled.Count * 0.9);
        var trainSet = shuffled.Take(trainSize).ToList();
        var valSet = shuffled.Skip(trainSize).ToList();
        
        // Calculate class balance
        int positives = trainSet.Count(s => s.answered);
        int negatives = trainSet.Count - positives;
        int totalBatches = (trainSet.Count + _config.BatchSize - 1) / _config.BatchSize;
        Console.WriteLine($"  Training samples: {trainSet.Count} ({positives} pos, {negatives} neg), Validation samples: {valSet.Count}");
        Console.WriteLine($"  Batches per epoch: {totalBatches}");

        float bestAuc = 0;
        int patienceCounter = 0;
        var trainingStartTime = DateTime.Now;

        for (int epoch = 0; epoch < _config.Epochs; epoch++)
        {
            var epochStartTime = DateTime.Now;

            // Shuffle training data
            trainSet = trainSet.OrderBy(_ => _random.Next()).ToList();

            // Train batches
            float epochLoss = 0;
            int numBatches = 0;

            for (int i = 0; i < trainSet.Count; i += _config.BatchSize)
            {
                var batch = trainSet.Skip(i).Take(_config.BatchSize).ToList();
                float batchLoss = _model.TrainBatch(batch);
                epochLoss += batchLoss;
                numBatches++;
            }

            epochLoss /= numBatches;
            result.TrainingLosses.Add(epochLoss);

            // Validate
            if (valSet.Count > 0)
            {
                var (auc, acc, f1) = _model.Evaluate(valSet);
                result.ValidationMetrics.Add((auc, acc, f1));

                // Track best AUC and early stopping (if enabled)
                if (auc > bestAuc)
                {
                    bestAuc = auc;
                    patienceCounter = 0;
                }
                else if (_config.EarlyStoppingPatience > 0)  // Only apply early stopping if patience > 0
                {
                    patienceCounter++;
                    if (patienceCounter >= _config.EarlyStoppingPatience)
                    {
                        Console.WriteLine($"  Early stopping at epoch {epoch + 1} (best AUC: {bestAuc:F4})");
                        break;
                    }
                }

                // Report progress every epoch for large datasets, every 5 epochs otherwise
                int reportInterval = trainSet.Count > 10000 ? 1 : 5;
                if ((epoch + 1) % reportInterval == 0 || epoch == 0)
                {
                    var epochTime = DateTime.Now - epochStartTime;
                    var totalElapsed = DateTime.Now - trainingStartTime;
                    var avgEpochTime = totalElapsed.TotalSeconds / (epoch + 1);
                    var remainingEpochs = _config.Epochs - epoch - 1;
                    var eta = TimeSpan.FromSeconds(avgEpochTime * remainingEpochs);

                    Console.WriteLine($"  Epoch {epoch + 1}/{_config.Epochs}: Loss={epochLoss:F4}, AUC={auc:F4}, ACC={acc:F4}, F1={f1:F4} ({epochTime.TotalSeconds:F1}s/epoch, ETA: {eta:mm\\:ss})");
                }
            }
        }
        
        stopwatch.Stop();
        result.TrainingTimeSeconds = stopwatch.Elapsed.TotalSeconds;
        
        Console.WriteLine($"\nTraining completed in {result.TrainingTimeSeconds:F1} seconds");
        
        return result;
    }
}

/// <summary>
/// Training configuration parameters.
/// </summary>
public class TrainingConfig
{
    // === Data Source Configuration ===

    /// <summary>
    /// Whether to use real StackOverflow data (true) or synthetic data (false).
    /// Default: false (synthetic data for testing)
    /// </summary>
    public bool UseRealData { get; set; } = false;

    /// <summary>
    /// Path to the StackOverflow dataset directory (when UseRealData = true).
    /// Expected structure: {DataPath}/knowledge_graph/, experts.json, questions.json, samples.json
    /// </summary>
    public string DataPath { get; set; } = "data/stackoverflow";

    /// <summary>
    /// Number of experts to generate (synthetic data only).
    /// </summary>
    public int NumExperts { get; set; } = 100;

    /// <summary>
    /// Number of questions to generate (synthetic data only).
    /// </summary>
    public int NumQuestions { get; set; } = 500;

    /// <summary>
    /// Number of training samples to generate (synthetic data only).
    /// </summary>
    public int NumSamples { get; set; } = 3000;

    // === Model Configuration ===

    /// <summary>
    /// Embedding dimension for all components (default: 64 for small data, 100 for large).
    /// </summary>
    public int EmbeddingDim { get; set; } = 64;

    /// <summary>
    /// SSL loss coefficient β (Equation 19: L = L(θ) + β * L_con).
    /// </summary>
    public float Beta { get; set; } = 0.1f;

    /// <summary>
    /// Enable hierarchical mutual information in SSL (recommended for large datasets only).
    /// </summary>
    public bool UseHierarchicalMI { get; set; } = false;

    // === Training Configuration ===

    public int Epochs { get; set; } = 100;
    public int BatchSize { get; set; } = 128;
    public float LearningRate { get; set; } = 1e-4f;
    public int TransHEpochs { get; set; } = 100;
    public int EPDRLEpisodes { get; set; } = 50;
    public int GCNEpochs { get; set; } = 50;
    public int EarlyStoppingPatience { get; set; } = 10;
    public int? Seed { get; set; } = 42;
}

/// <summary>
/// Training result containing losses and metrics.
/// </summary>
public class TrainingResult
{
    public List<float> TrainingLosses { get; set; } = new();
    public List<(float auc, float acc, float f1)> ValidationMetrics { get; set; } = new();
    public double TrainingTimeSeconds { get; set; }
}

