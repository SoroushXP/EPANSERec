using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace EPANSERec.Core.GraphNeuralNetworks;

/// <summary>
/// Graph Self-Supervised Learning module for expertise preference optimization.
/// Based on Deep Graph Infomax (DGI) approach described in the paper.
/// Maximizes mutual information between local and global representations.
/// </summary>
public class GraphSelfSupervisedLearning : Module<Tensor, Tensor, Tensor>
{
    private readonly Linear _discriminator;
    private readonly Linear _readout;
    private readonly float _gamma; // Coefficient to reduce gradient conflict
    private readonly int _embeddingDim;

    public GraphSelfSupervisedLearning(int embeddingDim, float gamma = 0.1f) 
        : base("GraphSelfSupervisedLearning")
    {
        _embeddingDim = embeddingDim;
        _gamma = gamma;
        
        // Discriminator for scoring consistency (Equation 12: fD function)
        _discriminator = Linear(embeddingDim * 2, 1);
        
        // Readout function for graph-level representation
        _readout = Linear(embeddingDim, embeddingDim);
        
        RegisterComponents();
    }

    /// <summary>
    /// Forward pass computing self-supervised contrastive loss.
    /// </summary>
    /// <param name="nodeEmbeddings">Node embeddings [N, embeddingDim]</param>
    /// <param name="adjacencyMatrix">Adjacency matrix [N, N]</param>
    /// <returns>Contrastive loss (Lcon from Equation 12)</returns>
    public override Tensor forward(Tensor nodeEmbeddings, Tensor adjacencyMatrix)
    {
        // Get graph-level representation using readout (Equation 11)
        var graphRepresentation = GetGraphRepresentation(nodeEmbeddings, adjacencyMatrix);
        
        // Create positive samples (actual node-graph pairs)
        var positiveLoss = ComputePositiveLoss(nodeEmbeddings, graphRepresentation);
        
        // Create negative samples (shuffled node-graph pairs)
        var negativeLoss = ComputeNegativeLoss(nodeEmbeddings, graphRepresentation);
        
        // Total contrastive loss (Equation 12)
        var totalLoss = positiveLoss + negativeLoss;
        
        return _gamma * totalLoss;
    }

    /// <summary>
    /// Computes graph-level representation using weighted sum (Equation 11).
    /// EL = γ * (a_ui / sum(a_ui))
    /// </summary>
    private Tensor GetGraphRepresentation(Tensor nodeEmbeddings, Tensor adjacencyMatrix)
    {
        // Weight nodes by their importance in adjacency
        var nodeWeights = adjacencyMatrix.sum(dim: 1, keepdim: true);
        var totalWeight = nodeWeights.sum() + 1e-10f;
        var normalizedWeights = nodeWeights / totalWeight;
        
        // Weighted sum of node embeddings
        var weightedSum = (nodeEmbeddings * normalizedWeights).sum(dim: 0, keepdim: true);
        
        // Apply readout transformation
        var graphRep = functional.sigmoid(_readout.forward(weightedSum));
        
        return graphRep;
    }

    /// <summary>
    /// Computes loss for positive samples (actual node-graph pairs).
    /// </summary>
    private Tensor ComputePositiveLoss(Tensor nodeEmbeddings, Tensor graphRepresentation)
    {
        int numNodes = (int)nodeEmbeddings.size(0);
        
        // Expand graph representation to match all nodes
        var expandedGraph = graphRepresentation.expand(numNodes, -1);
        
        // Concatenate node embeddings with graph representation
        var pairs = torch.cat(new[] { nodeEmbeddings, expandedGraph }, dim: 1);
        
        // Score using discriminator
        var scores = _discriminator.forward(pairs);
        var positiveScores = functional.sigmoid(scores);
        
        // Binary cross entropy loss for positive samples (label = 1)
        var loss = -torch.log(positiveScores + 1e-10f).mean();
        
        return loss;
    }

    /// <summary>
    /// Computes loss for negative samples (shuffled node-graph pairs).
    /// </summary>
    private Tensor ComputeNegativeLoss(Tensor nodeEmbeddings, Tensor graphRepresentation)
    {
        int numNodes = (int)nodeEmbeddings.size(0);
        
        // Shuffle node embeddings to create negative samples
        var indices = torch.randperm(numNodes, device: nodeEmbeddings.device);
        var shuffledNodes = nodeEmbeddings.index_select(0, indices);
        
        // Expand graph representation
        var expandedGraph = graphRepresentation.expand(numNodes, -1);
        
        // Concatenate shuffled nodes with graph representation
        var pairs = torch.cat(new[] { shuffledNodes, expandedGraph }, dim: 1);
        
        // Score using discriminator
        var scores = _discriminator.forward(pairs);
        var negativeScores = functional.sigmoid(scores);
        
        // Binary cross entropy loss for negative samples (label = 0)
        var loss = -torch.log(1 - negativeScores + 1e-10f).mean();
        
        return loss;
    }

    /// <summary>
    /// Samples positive subgraphs based on edge weights (for hierarchical mutual information).
    /// </summary>
    public Tensor SamplePositiveSubgraph(Tensor adjacencyMatrix, float threshold = 0.5f)
    {
        // Select edges with weight above threshold
        var mask = adjacencyMatrix > threshold;
        return adjacencyMatrix * mask.to_type(ScalarType.Float32);
    }
}

