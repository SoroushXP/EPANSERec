using EPANSERec.Core.KnowledgeGraph;

namespace EPANSERec.Core.Embeddings;

/// <summary>
/// Node2Vec implementation for learning graph structure embeddings.
/// Uses biased random walks with BFS/DFS exploration as described in the paper.
/// </summary>
public class Node2Vec
{
    private readonly SoftwareKnowledgeGraph _graph;
    private readonly int _embeddingDimension;
    private readonly int _walkLength;
    private readonly int _numWalks;
    private readonly double _p; // Return parameter (BFS-like if p > 1)
    private readonly double _q; // In-out parameter (DFS-like if q > 1)
    private readonly int _windowSize;
    private readonly Random _random;
    
    private Dictionary<int, float[]> _embeddings = new();

    public Node2Vec(
        SoftwareKnowledgeGraph graph,
        int embeddingDimension = 100,
        int walkLength = 80,
        int numWalks = 10,
        double p = 1.0,
        double q = 1.0,
        int windowSize = 5,
        int? seed = null)
    {
        _graph = graph;
        _embeddingDimension = embeddingDimension;
        _walkLength = walkLength;
        _numWalks = numWalks;
        _p = p;
        _q = q;
        _windowSize = windowSize;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Trains Node2Vec embeddings for all nodes in the graph.
    /// </summary>
    public Dictionary<int, float[]> Train(int epochs = 5, float learningRate = 0.025f)
    {
        // Initialize embeddings randomly
        InitializeEmbeddings();
        
        // Generate random walks
        var walks = GenerateWalks();
        
        // Train using Skip-gram model
        TrainSkipGram(walks, epochs, learningRate);
        
        return _embeddings;
    }

    private void InitializeEmbeddings()
    {
        foreach (var entityId in _graph.Entities.Keys)
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
    /// Trains embeddings using Skip-gram with negative sampling.
    /// </summary>
    private void TrainSkipGram(List<List<int>> walks, int epochs, float learningRate)
    {
        var nodes = _graph.Entities.Keys.ToList();
        int negSamples = 5;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            float totalLoss = 0;
            int count = 0;

            foreach (var walk in walks)
            {
                for (int i = 0; i < walk.Count; i++)
                {
                    var target = walk[i];

                    // Context window
                    int start = Math.Max(0, i - _windowSize);
                    int end = Math.Min(walk.Count - 1, i + _windowSize);

                    for (int j = start; j <= end; j++)
                    {
                        if (i == j) continue;

                        var context = walk[j];

                        // Positive sample update
                        totalLoss += UpdateEmbeddings(target, context, true, learningRate);
                        count++;

                        // Negative samples
                        for (int k = 0; k < negSamples; k++)
                        {
                            var negNode = nodes[_random.Next(nodes.Count)];
                            if (negNode != context && negNode != target)
                            {
                                totalLoss += UpdateEmbeddings(target, negNode, false, learningRate);
                            }
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Updates embeddings using SGD for a single pair.
    /// </summary>
    private float UpdateEmbeddings(int target, int context, bool isPositive, float lr)
    {
        var targetEmb = _embeddings[target];
        var contextEmb = _embeddings[context];

        // Compute dot product
        float dot = 0;
        for (int i = 0; i < _embeddingDimension; i++)
            dot += targetEmb[i] * contextEmb[i];

        // Sigmoid and gradient
        float sigmoid = 1.0f / (1.0f + (float)Math.Exp(-dot));
        float label = isPositive ? 1.0f : 0.0f;
        float gradient = lr * (label - sigmoid);

        // Update embeddings
        for (int i = 0; i < _embeddingDimension; i++)
        {
            float tempTarget = targetEmb[i];
            targetEmb[i] += gradient * contextEmb[i];
            contextEmb[i] += gradient * tempTarget;
        }

        // Return loss for monitoring
        return isPositive
            ? -(float)Math.Log(sigmoid + 1e-10)
            : -(float)Math.Log(1 - sigmoid + 1e-10);
    }

    /// <summary>
    /// Gets the embedding for a specific node.
    /// </summary>
    public float[] GetEmbedding(int nodeId) =>
        _embeddings.TryGetValue(nodeId, out var emb) ? emb : new float[_embeddingDimension];
}

