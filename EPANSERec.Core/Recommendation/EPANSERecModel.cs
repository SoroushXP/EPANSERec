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
/// </summary>
public class EPANSERecModel
{
    private readonly SoftwareKnowledgeGraph _knowledgeGraph;
    private readonly int _embeddingDim;
    private readonly float _beta; // SSL loss coefficient
    
    // Component modules
    private EPDRL? _epdrl;
    private TransH? _transH;
    private ExpertisePreferenceOptimizer? _preferenceOptimizer;
    private AttentionNetwork? _attentionNetwork;
    private FeatureFusion? _featureFusion;
    private PredictionDNN? _predictionDNN;
    private torch.optim.Optimizer? _optimizer;
    
    // Cached embeddings
    private Dictionary<int, float[]> _entityEmbeddings = new();
    private Dictionary<int, ExpertisePreferenceWeightGraph> _expertPreferenceGraphs = new();
    private Dictionary<int, float[]> _optimizedExpertEmbeddings = new();

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
        Console.WriteLine("  - Initializing EPDRL (Deep Reinforcement Learning)...");
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
        
        // 4. Initialize attention network
        Console.WriteLine("  - Initializing Attention Network...");
        _attentionNetwork = new AttentionNetwork(_embeddingDim);
        
        // 5. Initialize feature fusion
        Console.WriteLine("  - Initializing Feature Fusion...");
        _featureFusion = new FeatureFusion(_embeddingDim);
        
        // 6. Initialize prediction DNN
        Console.WriteLine("  - Initializing Prediction DNN...");
        // Input: question embedding + expert embedding
        _predictionDNN = new PredictionDNN(_embeddingDim * 2, hiddenDim: 256);
        
        // Combine trainable parameters
        var parameters = _attentionNetwork.parameters()
            .Concat(_featureFusion.parameters())
            .Concat(_predictionDNN.parameters());
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
        int count = 0;
        foreach (var expert in experts)
        {
            var preferenceGraph = _epdrl!.GeneratePreferenceGraph(expert, episodes);
            _expertPreferenceGraphs[expert.Id] = preferenceGraph;
            count++;
            if (count % 10 == 0)
                Console.WriteLine($"  Processed {count} experts...");
        }
        Console.WriteLine($"Generated {count} preference graphs.");
    }

    /// <summary>
    /// Optimizes expert embeddings using GCN with self-supervised learning.
    /// </summary>
    public void OptimizeExpertEmbeddings(int epochs = 50)
    {
        Console.WriteLine("Optimizing expert embeddings with GCN + SSL...");
        foreach (var (expertId, preferenceGraph) in _expertPreferenceGraphs)
        {
            var optimizedFeatures = _preferenceOptimizer!.OptimizeFeatures(
                preferenceGraph, _entityEmbeddings, epochs);
            
            // Store the mean of optimized features as expert embedding
            int numNodes = optimizedFeatures.GetLength(0);
            var expertEmb = new float[_embeddingDim];
            for (int i = 0; i < numNodes; i++)
                for (int j = 0; j < _embeddingDim; j++)
                    expertEmb[j] += optimizedFeatures[i, j] / numNodes;
            
            _optimizedExpertEmbeddings[expertId] = expertEmb;
        }
        Console.WriteLine($"Optimized {_optimizedExpertEmbeddings.Count} expert embeddings.");
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
    /// </summary>
    public float TrainBatch(List<(Question question, Expert expert, bool answered)> batch)
    {
        _optimizer!.zero_grad();

        var predictions = new List<Tensor>();
        var labels = new List<float>();

        foreach (var (question, expert, answered) in batch)
        {
            var questionEmb = GetQuestionEmbedding(question);
            var expertEmb = GetExpertEmbedding(expert, questionEmb);
            var input = torch.cat(new[] { questionEmb, expertEmb }, dim: -1).unsqueeze(0);
            var pred = _predictionDNN!.forward(input);
            predictions.Add(pred);
            labels.Add(answered ? 1.0f : 0.0f);
        }

        var predTensor = torch.cat(predictions.ToArray(), dim: 0);
        var labelTensor = torch.tensor(labels.ToArray()).unsqueeze(1);

        var loss = RecommendationLoss.ComputeBCELoss(predTensor, labelTensor);
        loss.backward();
        _optimizer.step();

        return loss.item<float>();
    }

    /// <summary>
    /// Gets question embedding by averaging entity embeddings.
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

        // Average pooling
        var avgEmb = new float[_embeddingDim];
        foreach (var emb in embeddings)
            for (int i = 0; i < _embeddingDim; i++)
                avgEmb[i] += emb[i] / embeddings.Count;

        return torch.tensor(avgEmb);
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
