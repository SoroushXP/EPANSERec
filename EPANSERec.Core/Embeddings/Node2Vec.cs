using EPANSERec.Core.KnowledgeGraph;

namespace EPANSERec.Core.Embeddings;

/// <summary>
/// Node2Vec implementation for learning graph structure embeddings.
/// Based on: Grover, A., & Leskovec, J. (2016). "node2vec: Scalable Feature Learning for Networks." KDD.
///
/// Uses biased random walks with BFS/DFS exploration controlled by parameters p and q.
/// The random walk strategy interpolates between BFS (exploring local neighborhoods) and
/// DFS (exploring distant nodes) based on:
/// - p (return parameter): Controls likelihood of immediately revisiting a node
/// - q (in-out parameter): Controls search to differentiate inward vs outward nodes
///
/// Training uses Skip-gram with Negative Sampling (SGNS) from Word2Vec
/// (Mikolov et al., 2013), treating node sequences as sentences.
///
/// Negative sampling uses noise distribution: P_n(v) ∝ d_v^0.75 (unigram distribution raised to 3/4 power)
/// as recommended in the original Word2Vec paper for better handling of rare words/nodes.
/// </summary>
public class Node2Vec
{
    private readonly SoftwareKnowledgeGraph _graph;
    private readonly int _embeddingDimension;
    private readonly int _walkLength;
    private readonly int _numWalks;
    private readonly double _p; // Return parameter: High p (>1) = less likely to return, low p (<1) = more BFS-like
    private readonly double _q; // In-out parameter: High q (>1) = biased toward local nodes (BFS), low q (<1) = biased toward distant nodes (DFS)
    private readonly int _windowSize;
    private readonly int _negSamples; // Number of negative samples per positive (typically 5-20)
    private readonly Random _random;

    private Dictionary<int, float[]> _embeddings = new();
    private List<int> _nodeList = new();
    private double[] _negSamplingDistribution = Array.Empty<double>(); // Precomputed unigram^0.75 distribution
    private double[] _cumulativeDistribution = Array.Empty<double>(); // For efficient sampling

    /// <summary>
    /// Initializes Node2Vec with configurable hyperparameters.
    /// </summary>
    /// <param name="graph">Knowledge graph to learn embeddings from</param>
    /// <param name="embeddingDimension">Dimension of output embeddings (default: 100, paper uses 128)</param>
    /// <param name="walkLength">Length of each random walk (default: 80, paper recommends 40-80)</param>
    /// <param name="numWalks">Number of walks per node (default: 10, paper uses 10)</param>
    /// <param name="p">Return parameter controlling BFS/DFS behavior (default: 1.0)</param>
    /// <param name="q">In-out parameter controlling BFS/DFS behavior (default: 1.0)</param>
    /// <param name="windowSize">Skip-gram context window size (default: 5, standard for Word2Vec)</param>
    /// <param name="negSamples">Number of negative samples per positive (default: 5, Word2Vec recommends 5-20)</param>
    /// <param name="seed">Random seed for reproducibility</param>
    public Node2Vec(
        SoftwareKnowledgeGraph graph,
        int embeddingDimension = 100,
        int walkLength = 80,
        int numWalks = 10,
        double p = 1.0,
        double q = 1.0,
        int windowSize = 5,
        int negSamples = 5,
        int? seed = null)
    {
        _graph = graph;
        _embeddingDimension = embeddingDimension;
        _walkLength = walkLength;
        _numWalks = numWalks;
        _p = p;
        _q = q;
        _windowSize = windowSize;
        _negSamples = negSamples;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Trains Node2Vec embeddings for all nodes in the graph.
    /// Training follows the two-phase approach from the original Node2Vec paper:
    /// 1. Generate corpus of random walks
    /// 2. Apply Skip-gram with Negative Sampling (SGNS)
    /// </summary>
    /// <param name="epochs">Number of training epochs over the walk corpus</param>
    /// <param name="learningRate">Initial learning rate (decays linearly during training)</param>
    public Dictionary<int, float[]> Train(int epochs = 5, float learningRate = 0.025f)
    {
        // Initialize embeddings randomly (Xavier/Glorot-like initialization)
        InitializeEmbeddings();

        // Build negative sampling distribution (unigram^0.75)
        BuildNegativeSamplingDistribution();

        // Generate random walks (corpus construction)
        var walks = GenerateWalks();

        // Train using Skip-gram with Negative Sampling (SGNS)
        TrainSkipGram(walks, epochs, learningRate);

        return _embeddings;
    }

    /// <summary>
    /// Initializes embeddings with small random values.
    /// Uses uniform distribution in [-0.5/d, 0.5/d] similar to Word2Vec default.
    /// </summary>
    private void InitializeEmbeddings()
    {
        _nodeList = _graph.Entities.Keys.ToList();
        foreach (var entityId in _nodeList)
        {
            var embedding = new float[_embeddingDimension];
            for (int i = 0; i < _embeddingDimension; i++)
            {
                embedding[i] = (float)(_random.NextDouble() - 0.5) / _embeddingDimension;
            }
            _embeddings[entityId] = embedding;
        }
    }

    /// <summary>
    /// Builds the noise distribution for negative sampling.
    /// Uses P_n(w) ∝ U(w)^0.75 where U(w) is the unigram (frequency) distribution.
    /// The 0.75 exponent helps sample rare nodes more frequently than their
    /// true frequency, improving embedding quality (Mikolov et al., 2013).
    /// </summary>
    private void BuildNegativeSamplingDistribution()
    {
        // Compute node degrees as proxy for frequency
        var degrees = new Dictionary<int, int>();
        foreach (var nodeId in _nodeList)
        {
            var neighbors = _graph.GetNeighbors(nodeId);
            degrees[nodeId] = Math.Max(1, neighbors.Count); // Minimum degree of 1
        }

        // Compute unigram^0.75 distribution
        _negSamplingDistribution = new double[_nodeList.Count];
        double total = 0;

        for (int i = 0; i < _nodeList.Count; i++)
        {
            // Raise degree to power 0.75 as per Word2Vec recommendation
            _negSamplingDistribution[i] = Math.Pow(degrees[_nodeList[i]], 0.75);
            total += _negSamplingDistribution[i];
        }

        // Normalize to probability distribution
        for (int i = 0; i < _nodeList.Count; i++)
        {
            _negSamplingDistribution[i] /= total;
        }

        // Build cumulative distribution for efficient sampling
        _cumulativeDistribution = new double[_nodeList.Count];
        _cumulativeDistribution[0] = _negSamplingDistribution[0];
        for (int i = 1; i < _nodeList.Count; i++)
        {
            _cumulativeDistribution[i] = _cumulativeDistribution[i - 1] + _negSamplingDistribution[i];
        }
    }

    /// <summary>
    /// Samples a node from the negative sampling distribution using binary search.
    /// </summary>
    private int SampleNegativeNode()
    {
        double r = _random.NextDouble();
        int low = 0, high = _cumulativeDistribution.Length - 1;

        while (low < high)
        {
            int mid = (low + high) / 2;
            if (_cumulativeDistribution[mid] < r)
                low = mid + 1;
            else
                high = mid;
        }

        return _nodeList[low];
    }

    /// <summary>
    /// Generates biased random walks from all nodes.
    /// </summary>
    private List<List<int>> GenerateWalks()
    {
        var walks = new List<List<int>>();
        var nodes = _graph.Entities.Keys.ToList();
        
        for (int i = 0; i < _numWalks; i++)
        {
            // Shuffle nodes for each iteration
            var shuffled = nodes.OrderBy(_ => _random.Next()).ToList();
            foreach (var node in shuffled)
            {
                walks.Add(BiasedRandomWalk(node));
            }
        }
        
        return walks;
    }

    /// <summary>
    /// Performs a biased random walk starting from a given node.
    /// </summary>
    private List<int> BiasedRandomWalk(int startNode)
    {
        var walk = new List<int> { startNode };
        
        while (walk.Count < _walkLength)
        {
            var current = walk[^1];
            var neighbors = _graph.GetNeighbors(current);
            
            if (neighbors.Count == 0) break;
            
            if (walk.Count == 1)
            {
                // First step: uniform random
                var idx = _random.Next(neighbors.Count);
                walk.Add(neighbors[idx].neighbor.Id);
            }
            else
            {
                // Biased step based on p and q
                var previous = walk[^2];
                var next = GetBiasedNeighbor(current, previous, neighbors);
                walk.Add(next);
            }
        }
        
        return walk;
    }

    /// <summary>
    /// Selects next node based on biased transition probabilities.
    /// </summary>
    private int GetBiasedNeighbor(int current, int previous, 
        List<(Entity neighbor, Relation relation, float weight)> neighbors)
    {
        var previousNeighbors = _graph.GetNeighbors(previous).Select(n => n.neighbor.Id).ToHashSet();
        var weights = new List<double>();
        
        foreach (var (neighbor, _, edgeWeight) in neighbors)
        {
            double alpha;
            if (neighbor.Id == previous)
                alpha = 1.0 / _p; // Return to previous node
            else if (previousNeighbors.Contains(neighbor.Id))
                alpha = 1.0; // Distance is 1 from previous
            else
                alpha = 1.0 / _q; // Distance is 2 from previous
            
            weights.Add(alpha * edgeWeight);
        }
        
        // Normalize and sample
        var total = weights.Sum();
        var r = _random.NextDouble() * total;
        var cumulative = 0.0;
        
        for (int i = 0; i < neighbors.Count; i++)
        {
            cumulative += weights[i];
            if (r <= cumulative)
                return neighbors[i].neighbor.Id;
        }
        
        return neighbors[^1].neighbor.Id;
    }

    /// <summary>
    /// Trains embeddings using Skip-gram with Negative Sampling (SGNS).
    /// Objective: maximize log σ(v_c · v_w) + Σ E[log σ(-v_n · v_w)]
    /// where v_w is target word/node, v_c is context, v_n are negative samples.
    ///
    /// Linear learning rate decay is applied as in original Word2Vec implementation.
    /// </summary>
    private void TrainSkipGram(List<List<int>> walks, int epochs, float initialLearningRate)
    {
        int totalIterations = epochs * walks.Count;
        int currentIteration = 0;
        float minLearningRate = initialLearningRate * 0.0001f; // Minimum LR

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Shuffle walks each epoch for better convergence
            var shuffledWalks = walks.OrderBy(_ => _random.Next()).ToList();

            foreach (var walk in shuffledWalks)
            {
                // Linear learning rate decay
                float progress = (float)currentIteration / totalIterations;
                float lr = Math.Max(minLearningRate, initialLearningRate * (1.0f - progress));
                currentIteration++;

                for (int i = 0; i < walk.Count; i++)
                {
                    var target = walk[i];

                    // Context window (dynamic window: sample from [1, windowSize])
                    int dynamicWindow = _random.Next(1, _windowSize + 1);
                    int start = Math.Max(0, i - dynamicWindow);
                    int end = Math.Min(walk.Count - 1, i + dynamicWindow);

                    for (int j = start; j <= end; j++)
                    {
                        if (i == j) continue;

                        var context = walk[j];

                        // Positive sample update: maximize log σ(v_c · v_w)
                        UpdateEmbeddings(target, context, true, lr);

                        // Negative samples from noise distribution P_n(w) ∝ U(w)^0.75
                        for (int k = 0; k < _negSamples; k++)
                        {
                            var negNode = SampleNegativeNode();
                            // Skip if negative sample is the same as context or target
                            if (negNode != context && negNode != target)
                            {
                                UpdateEmbeddings(target, negNode, false, lr);
                            }
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Updates embeddings using Stochastic Gradient Descent for a single (target, context) pair.
    ///
    /// For positive pairs: gradient = (1 - σ(v_c · v_w)) * v_c (maximize similarity)
    /// For negative pairs: gradient = -σ(v_n · v_w) * v_n (minimize similarity)
    ///
    /// Both target and context vectors are updated symmetrically.
    /// </summary>
    private void UpdateEmbeddings(int target, int context, bool isPositive, float lr)
    {
        var targetEmb = _embeddings[target];
        var contextEmb = _embeddings[context];

        // Compute dot product: v_target · v_context
        float dot = 0;
        for (int i = 0; i < _embeddingDimension; i++)
            dot += targetEmb[i] * contextEmb[i];

        // Clamp dot product to prevent overflow in sigmoid
        dot = Math.Clamp(dot, -6.0f, 6.0f);

        // Sigmoid activation: σ(x) = 1 / (1 + e^(-x))
        float sigmoid = 1.0f / (1.0f + (float)Math.Exp(-dot));

        // Gradient computation:
        // For positive: ∂log(σ(x))/∂x = 1 - σ(x)
        // For negative: ∂log(σ(-x))/∂x = -σ(x)
        float gradient = lr * (isPositive ? (1.0f - sigmoid) : -sigmoid);

        // Symmetric update of both embeddings
        for (int i = 0; i < _embeddingDimension; i++)
        {
            float tempTarget = targetEmb[i];
            targetEmb[i] += gradient * contextEmb[i];
            contextEmb[i] += gradient * tempTarget;
        }
    }

    /// <summary>
    /// Gets the embedding for a specific node.
    /// </summary>
    public float[] GetEmbedding(int nodeId) =>
        _embeddings.TryGetValue(nodeId, out var emb) ? emb : new float[_embeddingDimension];
}

