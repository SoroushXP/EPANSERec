using EPANSERec.Core.Embeddings;
using EPANSERec.Core.GraphNeuralNetworks;
using EPANSERec.Core.KnowledgeGraph;
using EPANSERec.Core.Models;
using EPANSERec.Core.ReinforcementLearning;
using TorchSharp;
using static TorchSharp.torch;

namespace EPANSERec.Core.Recommendation;

/// <summary>
/// EPAN-SERec: Expertise Preference-Aware Networks for Software Expert Recommendations.
/// Main model class that integrates all components from the paper.
///
/// Reference: Tang et al. (2024). "EPAN-SERec: Expertise preference-aware networks for
///            software expert recommendations with knowledge graph." Expert Systems with Applications.
///
/// ═══════════════════════════════════════════════════════════════════════════════════════════════
/// HYPERPARAMETER REFERENCE (from Paper Section 5.2 / Table 3)
/// ═══════════════════════════════════════════════════════════════════════════════════════════════
///
/// The following hyperparameters are used in the paper's experiments on StackOverflow data:
///
/// **Embedding Dimensions:**
///   - Entity/Node embedding dimension: 100 (can vary 50-200 per ablation study)
///   - GCN hidden dimension: 256
///
/// **Reinforcement Learning (EPDRL):**
///   - Max path length: 10
///   - γ (gamma) discount factor: 0.99 (standard DRL value)
///   - ε (epsilon) initial exploration: 1.0, decay to 0.01
///   - Replay buffer capacity: 10,000
///   - Batch size: 128
///
/// **Knowledge Graph Embedding (TransH):**
///   - Margin for ranking loss: 1.0
///   - Learning rate: 0.01
///
/// **Self-Supervised Learning:**
///   - β (beta) SSL loss coefficient: 0.1 (Equation 19: L = L(θ) + β * L_con)
///   - γ coefficient in SSL: 0.1 (reduces gradient conflict)
///
/// **Training:**
///   - Learning rate: 1e-4 (Adam optimizer)
///   - Epochs: 100 (early stopping based on validation)
///   - Dropout: 0.1-0.3 depending on component
/// ═══════════════════════════════════════════════════════════════════════════════════════════════
/// </summary>
public class EPANSERecModel
{
    private readonly SoftwareKnowledgeGraph _knowledgeGraph;
    private readonly int _embeddingDim;
    private readonly float _beta; // SSL loss coefficient β (Equation 19: L = L(θ) + β * L_con)

    // Component modules
    private EPDRL? _epdrl;
    private TransH? _transH;
    private ExpertisePreferenceOptimizer? _preferenceOptimizer;
    private AttentionNetwork? _attentionNetwork;
    private QuestionAttentionNetwork? _questionAttentionNetwork; // Self-attention for question embeddings
    private FeatureFusion? _featureFusion;
    private PredictionDNN? _predictionDNN;
    private GraphSelfSupervisedLearning? _sslModule; // SSL module for joint training
    private torch.optim.Optimizer? _optimizer;

    // Cached embeddings
    private Dictionary<int, float[]> _entityEmbeddings = new();
    private Dictionary<int, ExpertisePreferenceWeightGraph> _expertPreferenceGraphs = new();
    private Dictionary<int, float[]> _optimizedExpertEmbeddings = new();

    // Cached adjacency matrices for SSL during training (per expert)
    private Dictionary<int, Tensor> _expertAdjacencyMatrices = new();
    private Dictionary<int, Tensor> _expertFeatureMatrices = new();

    /// <summary>
    /// Creates the EPAN-SERec model with configurable hyperparameters.
    /// </summary>
    /// <param name="knowledgeGraph">Software knowledge graph from StackOverflow or similar</param>
    /// <param name="embeddingDim">Embedding dimension (paper default: 100, range: 50-200)</param>
    /// <param name="beta">SSL loss coefficient β (paper default: 0.1, Equation 19)</param>
    public EPANSERecModel(SoftwareKnowledgeGraph knowledgeGraph, int embeddingDim = 100, float beta = 0.1f)
    {
        _knowledgeGraph = knowledgeGraph;
        _embeddingDim = embeddingDim;
        _beta = beta;
    }

    /// <summary>
    /// Initializes all model components.
    /// </summary>
    public void Initialize(float learningRate = 1e-4f)
    {
        Console.WriteLine("Initializing EPAN-SERec model components...");

        // 1. Initialize EPDRL for expertise preference learning
        Console.WriteLine("  - Initializing EPDRL (Double DQN for Reinforcement Learning)...");
        _epdrl = new EPDRL(_knowledgeGraph, _embeddingDim);

        // 2. Initialize TransH for knowledge graph embeddings
        Console.WriteLine("  - Initializing TransH (Knowledge Graph Embedding)...");
        int numEntities = _knowledgeGraph.Entities.Count;
        int numRelations = _knowledgeGraph.Relations.Count;
        _transH = new TransH(numEntities, numRelations, _embeddingDim);

        // 3. Initialize GCN with self-supervised learning
        Console.WriteLine("  - Initializing Expertise Preference Optimizer (GCN + SSL)...");
        _preferenceOptimizer = new ExpertisePreferenceOptimizer(
            inputDim: _embeddingDim,
            hiddenDim: 256,
            outputDim: _embeddingDim);

        // 4. Initialize attention networks
        Console.WriteLine("  - Initializing Attention Networks...");
        _attentionNetwork = new AttentionNetwork(_embeddingDim);
        _questionAttentionNetwork = new QuestionAttentionNetwork(_embeddingDim);

        // 5. Initialize feature fusion (with gating mechanism)
        Console.WriteLine("  - Initializing Feature Fusion (Gating)...");
        _featureFusion = new FeatureFusion(_embeddingDim);

        // 6. Initialize prediction DNN
        Console.WriteLine("  - Initializing Prediction DNN...");
        // Input: question embedding + expert embedding
        _predictionDNN = new PredictionDNN(_embeddingDim * 2, hiddenDim: 256);

        // 7. Initialize SSL module for joint training (Equation 12 & 19)
        Console.WriteLine("  - Initializing SSL Module for Joint Training...");
        _sslModule = new GraphSelfSupervisedLearning(_embeddingDim, gamma: _beta);

        // Combine trainable parameters from all modules
        var parameters = _attentionNetwork.parameters()
            .Concat(_questionAttentionNetwork.parameters())
            .Concat(_featureFusion.parameters())
            .Concat(_predictionDNN.parameters())
            .Concat(_sslModule.parameters());
        _optimizer = torch.optim.Adam(parameters, lr: learningRate);

        Console.WriteLine("Model initialization complete.");
    }

    /// <summary>
    /// Pre-trains knowledge graph embeddings using TransH.
    /// </summary>
    public void PretrainKnowledgeEmbeddings(int epochs = 100, int batchSize = 128)
    {
        Console.WriteLine($"Pre-training TransH embeddings for {epochs} epochs...");
        _transH!.Train(_knowledgeGraph, epochs, batchSize);
        
        // Cache entity embeddings
        _entityEmbeddings = _transH.GetAllEntityEmbeddings(_knowledgeGraph.Entities.Keys);
        Console.WriteLine($"Cached {_entityEmbeddings.Count} entity embeddings.");
    }

    /// <summary>
    /// Generates expertise preference graphs for all experts using EPDRL.
    /// </summary>
    public void GenerateExpertPreferenceGraphs(IEnumerable<Expert> experts, int episodes = 100)
    {
        Console.WriteLine("Generating expertise preference graphs...");
        var expertList = experts.ToList();
        int totalExperts = expertList.Count;
        int count = 0;
        var startTime = DateTime.Now;
        Console.Out.Flush();

        // Use thread-safe dictionary for parallel execution
        var results = new System.Collections.Concurrent.ConcurrentDictionary<int, ExpertisePreferenceWeightGraph>();

        // Determine parallelism - use up to 75% of available cores
        int maxParallelism = Math.Max(1, Environment.ProcessorCount * 3 / 4);
        Console.WriteLine($"  Using {maxParallelism} parallel workers...");

        var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = maxParallelism };

        Parallel.ForEach(expertList, parallelOptions, expert =>
        {
            // Each thread needs its own EPDRL instance for thread safety
            var localEpdrl = new EPDRL(_knowledgeGraph, _embeddingDim);
            var preferenceGraph = localEpdrl.GeneratePreferenceGraph(expert, episodes);
            results[expert.Id] = preferenceGraph;

            var currentCount = Interlocked.Increment(ref count);

            // Progress reporting - thread-safe, less frequent to avoid console contention
            if (currentCount == 1 || currentCount % 100 == 0 || currentCount == totalExperts)
            {
                var elapsed = DateTime.Now - startTime;
                var avgTimePerExpert = elapsed.TotalSeconds / currentCount;
                var remaining = TimeSpan.FromSeconds(avgTimePerExpert * (totalExperts - currentCount));
                Console.WriteLine($"  Generated {currentCount}/{totalExperts} graphs ({100.0 * currentCount / totalExperts:F1}%) - ETA: {remaining:mm\\:ss}");
                Console.Out.Flush();
            }
        });

        // Copy results to main dictionary
        foreach (var kvp in results)
        {
            _expertPreferenceGraphs[kvp.Key] = kvp.Value;
        }

        Console.WriteLine($"Generated {count} preference graphs.");
    }

    /// <summary>
    /// Optimizes expert embeddings using GCN with self-supervised learning.
    /// Also caches adjacency and feature matrices for joint SSL training.
    /// </summary>
    public void OptimizeExpertEmbeddings(int epochs = 50)
    {
        Console.WriteLine("Optimizing expert embeddings with GCN + SSL...");
        int totalExperts = _expertPreferenceGraphs.Count;
        int processed = 0;
        var startTime = DateTime.Now;

        // Show initial message so user knows it's working
        Console.WriteLine($"  Processing {totalExperts} experts ({epochs} GCN epochs each)...");
        Console.Out.Flush();

        foreach (var (expertId, preferenceGraph) in _expertPreferenceGraphs)
        {
            var optimizedFeatures = _preferenceOptimizer!.OptimizeFeatures(
                preferenceGraph, _entityEmbeddings, epochs);

            // Store the mean of optimized features as expert embedding
            int numNodes = optimizedFeatures.GetLength(0);
            var expertEmb = new float[_embeddingDim];

            if (numNodes > 0)
            {
                for (int i = 0; i < numNodes; i++)
                    for (int j = 0; j < _embeddingDim; j++)
                        expertEmb[j] += optimizedFeatures[i, j] / numNodes;
            }
            // else: expertEmb stays as zeros for experts with no preference graph

            _optimizedExpertEmbeddings[expertId] = expertEmb;

            // Cache adjacency matrix and feature matrix for joint SSL training
            CacheExpertMatrices(expertId, preferenceGraph, optimizedFeatures);

            // Progress reporting - more frequent at start, then every 50
            processed++;
            bool shouldReport = processed == 1 || processed == 5 || processed == 10 ||
                               processed % 50 == 0 || processed == totalExperts;
            if (shouldReport)
            {
                var elapsed = DateTime.Now - startTime;
                var avgTimePerExpert = elapsed.TotalSeconds / processed;
                var remaining = TimeSpan.FromSeconds(avgTimePerExpert * (totalExperts - processed));
                Console.WriteLine($"  Optimized {processed}/{totalExperts} experts ({100.0 * processed / totalExperts:F1}%) - {avgTimePerExpert:F2}s/expert - ETA: {remaining:mm\\:ss}");
                Console.Out.Flush();
            }
        }
        Console.WriteLine($"Optimized {_optimizedExpertEmbeddings.Count} expert embeddings.");
        Console.WriteLine($"Cached {_expertAdjacencyMatrices.Count} adjacency matrices for joint SSL training.");
    }

    /// <summary>
    /// Caches adjacency and feature matrices for an expert for use in joint SSL training.
    /// </summary>
    private void CacheExpertMatrices(int expertId, ExpertisePreferenceWeightGraph preferenceGraph, float[,] features)
    {
        // Get weighted adjacency matrix
        var adjMatrix = preferenceGraph.GetWeightedAdjacencyMatrix();
        int n = adjMatrix.GetLength(0);

        if (n == 0) return;

        // Convert to flat arrays for tensor creation
        var adjArray = new float[n * n];
        var featureArray = new float[n * _embeddingDim];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                adjArray[i * n + j] = adjMatrix[i, j];

            for (int j = 0; j < _embeddingDim; j++)
                featureArray[i * _embeddingDim + j] = features[i, j];
        }

        // Store as tensors (detached, no gradient tracking for storage)
        _expertAdjacencyMatrices[expertId] = torch.tensor(adjArray).reshape(n, n).detach();
        _expertFeatureMatrices[expertId] = torch.tensor(featureArray).reshape(n, _embeddingDim).detach();
    }

    /// <summary>
    /// Predicts the probability of an expert answering a question.
    /// </summary>
    public float Predict(Question question, Expert expert)
    {
        using (torch.no_grad())
        {
            // Get question embedding from TransH
            var questionEmb = GetQuestionEmbedding(question);

            // Get expert embedding (attention-weighted historical Q&A)
            var expertEmb = GetExpertEmbedding(expert, questionEmb);

            // Concatenate and predict
            var input = torch.cat(new[] { questionEmb, expertEmb }, dim: -1).unsqueeze(0);
            var prediction = _predictionDNN!.forward(input);

            return prediction.item<float>();
        }
    }

    /// <summary>
    /// Recommends top-K experts for a question.
    /// </summary>
    public List<(Expert expert, float score)> RecommendExperts(Question question,
        IEnumerable<Expert> candidateExperts, int topK = 10)
    {
        var scores = new List<(Expert expert, float score)>();

        foreach (var expert in candidateExperts)
        {
            float score = Predict(question, expert);
            scores.Add((expert, score));
        }

        return scores.OrderByDescending(x => x.score).Take(topK).ToList();
    }

    /// <summary>
    /// Trains the model on a batch of (question, expert, label) samples.
    /// Implements Equation 19: L = L(θ) + β * L_con (joint BCE + SSL loss).
    /// </summary>
    public float TrainBatch(List<(Question question, Expert expert, bool answered)> batch)
    {
        _optimizer!.zero_grad();

        var predictions = new List<Tensor>();
        var labels = new List<float>();
        var expertIdsInBatch = new HashSet<int>();

        foreach (var (question, expert, answered) in batch)
        {
            var questionEmb = GetQuestionEmbedding(question);
            var expertEmb = GetExpertEmbedding(expert, questionEmb);
            var input = torch.cat(new[] { questionEmb, expertEmb }, dim: -1).unsqueeze(0);
            var pred = _predictionDNN!.forward(input);
            predictions.Add(pred);
            labels.Add(answered ? 1.0f : 0.0f);
            expertIdsInBatch.Add(expert.Id);
        }

        var predTensor = torch.cat(predictions.ToArray(), dim: 0);
        var labelTensor = torch.tensor(labels.ToArray()).unsqueeze(1);

        // Compute BCE loss (Equation 18)
        var bceLoss = RecommendationLoss.ComputeBCELoss(predTensor, labelTensor);

        // Compute SSL loss for experts in this batch (Equation 12)
        var sslLoss = ComputeBatchSSLLoss(expertIdsInBatch);

        // Joint loss: L = L(θ) + β * L_con (Equation 19)
        // Note: _beta is already applied inside the SSL module, so we use it directly
        var totalLoss = bceLoss + sslLoss;

        totalLoss.backward();
        _optimizer.step();

        return totalLoss.item<float>();
    }

    /// <summary>
    /// Computes the self-supervised contrastive loss for experts in the batch.
    /// Implements L_con from Equation 12.
    /// </summary>
    private Tensor ComputeBatchSSLLoss(HashSet<int> expertIds)
    {
        var sslLosses = new List<Tensor>();

        foreach (var expertId in expertIds)
        {
            // Check if we have cached matrices for this expert
            if (_expertAdjacencyMatrices.TryGetValue(expertId, out var adjMatrix) &&
                _expertFeatureMatrices.TryGetValue(expertId, out var featureMatrix))
            {
                // Compute SSL loss for this expert's preference graph
                // The SSL module returns β * L_con (gamma is set to _beta)
                var expertSslLoss = _sslModule!.forward(featureMatrix, adjMatrix);
                sslLosses.Add(expertSslLoss);
            }
        }

        if (sslLosses.Count == 0)
        {
            // Return zero loss if no SSL data available
            return torch.tensor(0.0f);
        }

        // Average SSL loss across experts in batch
        var stackedLosses = torch.stack(sslLosses.ToArray());
        return stackedLosses.mean();
    }

    /// <summary>
    /// Gets question embedding using self-attention over entity embeddings.
    /// Uses QuestionAttentionNetwork for attention-weighted aggregation instead of simple mean pooling.
    /// </summary>
    private Tensor GetQuestionEmbedding(Question question)
    {
        var embeddings = new List<float[]>();
        foreach (var entityId in question.EntityIds)
        {
            if (_entityEmbeddings.TryGetValue(entityId, out var emb))
                embeddings.Add(emb);
        }

        if (embeddings.Count == 0)
            return torch.zeros(_embeddingDim);

        // Convert to tensor for attention computation
        var embTensor = torch.tensor(embeddings.SelectMany(e => e).ToArray())
            .reshape(embeddings.Count, _embeddingDim);

        // Use self-attention to compute question embedding
        return _questionAttentionNetwork!.forward(embTensor);
    }

    /// <summary>
    /// Gets expert embedding using attention over historical Q&A.
    /// </summary>
    private Tensor GetExpertEmbedding(Expert expert, Tensor questionEmbedding)
    {
        // If we have optimized embedding, use it
        if (_optimizedExpertEmbeddings.TryGetValue(expert.Id, out var optimizedEmb))
        {
            var optimizedTensor = torch.tensor(optimizedEmb);

            // Fuse with semantic embedding if available
            if (_entityEmbeddings.TryGetValue(expert.Id, out var semanticEmb))
            {
                var semanticTensor = torch.tensor(semanticEmb);
                return _featureFusion!.forward(optimizedTensor, semanticTensor);
            }
            return optimizedTensor;
        }

        // Fallback: use attention over historical Q&A embeddings
        var historicalEmbs = new List<float[]>();
        foreach (var entityId in expert.HistoricalEntityIds)
        {
            if (_entityEmbeddings.TryGetValue(entityId, out var emb))
                historicalEmbs.Add(emb);
        }

        if (historicalEmbs.Count == 0)
            return torch.zeros(_embeddingDim);

        var histTensor = torch.tensor(historicalEmbs.SelectMany(e => e).ToArray())
            .reshape(historicalEmbs.Count, _embeddingDim);

        return _attentionNetwork!.forward(questionEmbedding, histTensor);
    }

    /// <summary>
    /// Evaluates model performance on test data.
    /// </summary>
    public (float auc, float accuracy, float f1) Evaluate(
        List<(Question question, Expert expert, bool answered)> testData)
    {
        var predictions = new List<float>();
        var labels = new List<int>();

        foreach (var (question, expert, answered) in testData)
        {
            float pred = Predict(question, expert);
            predictions.Add(pred);
            labels.Add(answered ? 1 : 0);
        }

        return ComputeMetrics(predictions, labels);
    }

    private (float auc, float accuracy, float f1) ComputeMetrics(List<float> predictions, List<int> labels)
    {
        // Compute accuracy
        int correct = 0;
        int tp = 0, fp = 0, fn = 0;

        for (int i = 0; i < predictions.Count; i++)
        {
            int predicted = predictions[i] >= 0.5f ? 1 : 0;
            if (predicted == labels[i]) correct++;
            if (predicted == 1 && labels[i] == 1) tp++;
            if (predicted == 1 && labels[i] == 0) fp++;
            if (predicted == 0 && labels[i] == 1) fn++;
        }

        float accuracy = (float)correct / predictions.Count;
        float precision = tp + fp > 0 ? (float)tp / (tp + fp) : 0;
        float recall = tp + fn > 0 ? (float)tp / (tp + fn) : 0;
        float f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;

        // Simplified AUC calculation
        float auc = ComputeAUC(predictions, labels);

        return (auc, accuracy, f1);
    }

    private float ComputeAUC(List<float> predictions, List<int> labels)
    {
        var pairs = predictions.Zip(labels, (p, l) => (pred: p, label: l))
            .OrderByDescending(x => x.pred).ToList();

        int positives = labels.Count(l => l == 1);
        int negatives = labels.Count - positives;

        if (positives == 0 || negatives == 0) return 0.5f;

        float auc = 0;
        int tpCount = 0;

        foreach (var (pred, label) in pairs)
        {
            if (label == 1) tpCount++;
            else auc += tpCount;
        }

        return auc / (positives * negatives);
    }
}
