using EPANSERec.Core.Models;
using TorchSharp;
using static TorchSharp.torch;

namespace EPANSERec.Core.GraphNeuralNetworks;

/// <summary>
/// Expertise Preference Optimization module that combines GCN with self-supervised learning.
/// Section 4.2 of the paper.
/// </summary>
public class ExpertisePreferenceOptimizer
{
    private readonly ExpertiseGCN _gcn;
    private readonly GraphSelfSupervisedLearning _ssl;
    private readonly torch.optim.Optimizer _optimizer;
    private readonly int _embeddingDim;
    private readonly float _sslWeight; // β coefficient for SSL loss

    public ExpertisePreferenceOptimizer(
        int inputDim,
        int hiddenDim = 256,
        int outputDim = 100,
        int numGcnLayers = 2,
        float learningRate = 1e-4f,
        float sslWeight = 0.1f,
        float dropoutRate = 0.1f)
    {
        _embeddingDim = outputDim;
        _sslWeight = sslWeight;
        
        _gcn = new ExpertiseGCN(inputDim, hiddenDim, outputDim, numGcnLayers, dropoutRate);
        _ssl = new GraphSelfSupervisedLearning(outputDim);
        
        // Combine parameters from both modules
        var parameters = _gcn.parameters().Concat(_ssl.parameters());
        _optimizer = torch.optim.Adam(parameters, lr: learningRate);
    }

    /// <summary>
    /// Optimizes expertise preference features using GCN and self-supervised learning.
    /// </summary>
    /// <param name="preferenceGraph">Expert's preference weight graph</param>
    /// <param name="initialFeatures">Initial node features from Node2Vec</param>
    /// <param name="epochs">Number of training epochs</param>
    /// <returns>Optimized node embeddings</returns>
    public float[,] OptimizeFeatures(
        ExpertisePreferenceWeightGraph preferenceGraph,
        Dictionary<int, float[]> initialFeatures,
        int epochs = 50)
    {
        // Handle empty preference graphs
        if (preferenceGraph.EntityIds.Count == 0)
        {
            return new float[0, _embeddingDim];
        }

        // Convert to tensors
        var (nodeFeatures, adjacencyMatrix, entityIdToIndex) = PrepareInputs(preferenceGraph, initialFeatures);

        // Handle single-node graphs (no edges to learn from)
        if (entityIdToIndex.Count == 1)
        {
            using (torch.no_grad())
            {
                var singleNodeOutput = _gcn.forward(nodeFeatures, adjacencyMatrix);
                return ConvertToArray(singleNodeOutput, 1);
            }
        }

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            _optimizer.zero_grad();

            // Forward pass through GCN
            var gcnOutput = _gcn.forward(nodeFeatures, adjacencyMatrix);

            // Self-supervised loss (Equation 12)
            var sslLoss = _ssl.forward(gcnOutput, adjacencyMatrix);

            // Total loss with SSL regularization
            var totalLoss = _sslWeight * sslLoss;

            totalLoss.backward();
            _optimizer.step();
        }

        // Get final embeddings
        using (torch.no_grad())
        {
            var finalEmbeddings = _gcn.forward(nodeFeatures, adjacencyMatrix);
            return ConvertToArray(finalEmbeddings, entityIdToIndex.Count);
        }
    }

    /// <summary>
    /// Prepares input tensors from preference graph.
    /// </summary>
    private (Tensor features, Tensor adjacency, Dictionary<int, int> idToIndex) PrepareInputs(
        ExpertisePreferenceWeightGraph preferenceGraph,
        Dictionary<int, float[]> initialFeatures)
    {
        var entityIds = preferenceGraph.EntityIds.OrderBy(x => x).ToList();
        var idToIndex = entityIds.Select((id, idx) => (id, idx)).ToDictionary(x => x.id, x => x.idx);
        int n = entityIds.Count;
        
        // Prepare node features matrix
        int featureDim = initialFeatures.Values.FirstOrDefault()?.Length ?? _embeddingDim;
        var featuresArray = new float[n * featureDim];
        
        for (int i = 0; i < n; i++)
        {
            var entityId = entityIds[i];
            var features = initialFeatures.GetValueOrDefault(entityId) ?? new float[featureDim];
            Array.Copy(features, 0, featuresArray, i * featureDim, featureDim);
        }
        
        // Get weighted adjacency matrix
        var adjMatrix = preferenceGraph.GetWeightedAdjacencyMatrix();
        var adjArray = new float[n * n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                adjArray[i * n + j] = adjMatrix[i, j];
        
        var featuresTensor = torch.tensor(featuresArray).reshape(n, featureDim);
        var adjacencyTensor = torch.tensor(adjArray).reshape(n, n);
        
        return (featuresTensor, adjacencyTensor, idToIndex);
    }

    /// <summary>
    /// Converts tensor to 2D float array.
    /// </summary>
    private float[,] ConvertToArray(Tensor tensor, int numNodes)
    {
        int embDim = (int)tensor.size(1);
        var result = new float[numNodes, embDim];
        var data = tensor.data<float>().ToArray();
        
        for (int i = 0; i < numNodes; i++)
            for (int j = 0; j < embDim; j++)
                result[i, j] = data[i * embDim + j];
        
        return result;
    }

    /// <summary>
    /// Gets the optimized embedding for a specific entity.
    /// </summary>
    public float[] GetOptimizedEmbedding(float[,] optimizedFeatures, int index)
    {
        int embDim = optimizedFeatures.GetLength(1);
        var result = new float[embDim];
        for (int j = 0; j < embDim; j++)
            result[j] = optimizedFeatures[index, j];
        return result;
    }
}

