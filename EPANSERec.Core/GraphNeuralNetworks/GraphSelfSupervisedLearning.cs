using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace EPANSERec.Core.GraphNeuralNetworks;

/// <summary>
/// Graph Self-Supervised Learning module for expertise preference optimization.
///
/// Implements the hierarchical mutual information maximization approach from the EPAN-SERec paper
/// (Section 4.2, Equations 11-12).
///
/// The module uses a contrastive learning approach inspired by Deep Graph Infomax (DGI),
/// maximizing mutual information at two levels:
///
/// 1. **Local-Global MI**: Between node representations and graph-level summary
///    - Positive pairs: (node_i, graph_summary) where node_i belongs to the graph
///    - Negative pairs: (shuffled_node_j, graph_summary) creating corrupted views
///
/// 2. **Hierarchical MI** (via subgraph sampling): Between nodes and weighted subgraphs
///    - Uses expertise preference weights to identify semantically coherent subgraphs
///    - High-weight edges indicate strong expertise preference relationships
///    - Subgraph-level summaries capture localized expertise patterns
///
/// Loss function (Equation 12):
/// L_con = -E[log(fD(h_i, s_G))] - E[log(1 - fD(h_j, s_G))]
///
/// where fD is the discriminator, h_i are node embeddings, and s_G is graph summary.
/// The coefficient γ (gamma) reduces gradient conflict between SSL and main task losses.
/// </summary>
public class GraphSelfSupervisedLearning : Module<Tensor, Tensor, Tensor>
{
    private readonly Linear _discriminator;           // Bilinear discriminator for MI estimation
    private readonly Linear? _subgraphDiscriminator;  // Discriminator for hierarchical MI (optional)
    private readonly Linear _readout;                 // Graph-level readout function
    private readonly Linear? _subgraphReadout;        // Subgraph-level readout function (optional)
    private readonly float _gamma;                    // SSL loss coefficient (typically 0.1)
    private readonly float _hierarchicalWeight;       // Weight for hierarchical MI term
    private readonly float _subgraphThreshold;        // Threshold for subgraph sampling
    private readonly bool _useHierarchicalMI;         // Whether to use hierarchical MI (disabled for small datasets)
    private readonly int _embeddingDim;

    /// <summary>
    /// Initializes the SSL module with configurable parameters.
    /// </summary>
    /// <param name="embeddingDim">Dimension of node embeddings</param>
    /// <param name="gamma">SSL loss coefficient β from Equation 19 (default: 0.1)</param>
    /// <param name="useHierarchicalMI">Enable hierarchical MI for large datasets (default: false)</param>
    /// <param name="hierarchicalWeight">Weight for hierarchical MI term (default: 0.5)</param>
    /// <param name="subgraphThreshold">Edge weight threshold for subgraph sampling (default: 0.3)</param>
    public GraphSelfSupervisedLearning(
        int embeddingDim,
        float gamma = 0.1f,
        bool useHierarchicalMI = false,
        float hierarchicalWeight = 0.5f,
        float subgraphThreshold = 0.3f)
        : base("GraphSelfSupervisedLearning")
    {
        _embeddingDim = embeddingDim;
        _gamma = gamma;
        _useHierarchicalMI = useHierarchicalMI;
        _hierarchicalWeight = hierarchicalWeight;
        _subgraphThreshold = subgraphThreshold;

        // Main discriminator for scoring node-graph consistency (Equation 12: fD function)
        // Uses bilinear scoring: fD(h, s) = σ(h^T W s)
        _discriminator = Linear(embeddingDim * 2, 1);

        // Readout function for graph-level representation (Equation 11)
        // R: h_V → s_G where s_G summarizes the entire graph
        _readout = Linear(embeddingDim, embeddingDim);

        // Only create hierarchical components if enabled (for large datasets)
        if (_useHierarchicalMI)
        {
            // Discriminator for hierarchical subgraph MI
            _subgraphDiscriminator = Linear(embeddingDim * 2, 1);

            // Readout for subgraph-level representation
            _subgraphReadout = Linear(embeddingDim, embeddingDim);
        }

        RegisterComponents();
    }

    /// <summary>
    /// Forward pass computing self-supervised contrastive loss.
    ///
    /// When hierarchical MI is disabled (default for small datasets):
    ///   L_con = L_global (standard DGI approach)
    ///
    /// When hierarchical MI is enabled (for large datasets):
    ///   L_con = L_global + α * L_hierarchical
    /// </summary>
    /// <param name="nodeEmbeddings">Node embeddings [N, embeddingDim]</param>
    /// <param name="adjacencyMatrix">Weighted adjacency matrix [N, N] (expertise preference weights)</param>
    /// <returns>Scaled contrastive loss: γ * L_con</returns>
    public override Tensor forward(Tensor nodeEmbeddings, Tensor adjacencyMatrix)
    {
        // 1. Global-level contrastive loss (standard DGI approach)
        var graphRepresentation = GetGraphRepresentation(nodeEmbeddings, adjacencyMatrix, _readout);
        var globalPositiveLoss = ComputePositiveLoss(nodeEmbeddings, graphRepresentation, _discriminator);
        var globalNegativeLoss = ComputeNegativeLoss(nodeEmbeddings, graphRepresentation, _discriminator);
        var globalLoss = globalPositiveLoss + globalNegativeLoss;

        // 2. Hierarchical MI loss (only if enabled - for large datasets with meaningful subgraph structure)
        Tensor totalLoss;

        if (_useHierarchicalMI && _subgraphDiscriminator != null && _subgraphReadout != null)
        {
            // Sample high-weight subgraph representing strong expertise preferences
            var subgraphAdj = SamplePositiveSubgraph(adjacencyMatrix, _subgraphThreshold);
            var subgraphEdgeCount = subgraphAdj.sum().item<float>();

            Tensor hierarchicalLoss;
            if (subgraphEdgeCount > 1.0f)
            {
                var subgraphRepresentation = GetGraphRepresentation(nodeEmbeddings, subgraphAdj, _subgraphReadout);
                var subgraphPositiveLoss = ComputePositiveLoss(nodeEmbeddings, subgraphRepresentation, _subgraphDiscriminator);
                var subgraphNegativeLoss = ComputeNegativeLoss(nodeEmbeddings, subgraphRepresentation, _subgraphDiscriminator);
                hierarchicalLoss = subgraphPositiveLoss + subgraphNegativeLoss;
            }
            else
            {
                hierarchicalLoss = torch.tensor(0.0f);
            }

            // Combined loss: L_con = L_global + α * L_hierarchical (Equation 12 extended)
            totalLoss = globalLoss + _hierarchicalWeight * hierarchicalLoss;
        }
        else
        {
            // Simple global MI only (faster, better for small datasets)
            totalLoss = globalLoss;
        }

        // Apply γ coefficient to reduce gradient conflict (Equation 19)
        return _gamma * totalLoss;
    }

    /// <summary>
    /// Computes graph/subgraph-level representation using weighted sum (Equation 11).
    ///
    /// s_G = σ(W * Σ(α_i * h_i))
    ///
    /// where α_i = a_ui / Σ(a_uj) represents the normalized importance weight
    /// based on edge weights (expertise preference strength).
    /// </summary>
    private Tensor GetGraphRepresentation(Tensor nodeEmbeddings, Tensor adjacencyMatrix, Linear readoutLayer)
    {
        // Compute node importance weights from adjacency (degree-based attention)
        var nodeWeights = adjacencyMatrix.sum(dim: 1, keepdim: true);
        var totalWeight = nodeWeights.sum() + 1e-10f;
        var normalizedWeights = nodeWeights / totalWeight;

        // Weighted sum of node embeddings: Σ(α_i * h_i)
        var weightedSum = (nodeEmbeddings * normalizedWeights).sum(dim: 0, keepdim: true);

        // Apply readout transformation with sigmoid activation: σ(W * x)
        var graphRep = functional.sigmoid(readoutLayer.forward(weightedSum));

        return graphRep;
    }

    /// <summary>
    /// Computes loss for positive samples (actual node-graph pairs).
    ///
    /// L_pos = -E[log(fD(h_i, s_G))]
    ///
    /// Maximizes mutual information between real node embeddings and graph summary.
    /// </summary>
    private Tensor ComputePositiveLoss(Tensor nodeEmbeddings, Tensor graphRepresentation, Linear discriminator)
    {
        int numNodes = (int)nodeEmbeddings.size(0);

        // Expand graph representation to match all nodes
        var expandedGraph = graphRepresentation.expand(numNodes, -1);

        // Concatenate [h_i; s_G] for discriminator input
        var pairs = torch.cat(new[] { nodeEmbeddings, expandedGraph }, dim: 1);

        // Score using discriminator: fD(h_i, s_G)
        var scores = discriminator.forward(pairs);
        var positiveScores = functional.sigmoid(scores);

        // BCE loss for positive samples: -log(σ(fD(h_i, s_G)))
        var loss = -torch.log(positiveScores + 1e-10f).mean();

        return loss;
    }

    /// <summary>
    /// Computes loss for negative samples (shuffled/corrupted node-graph pairs).
    ///
    /// L_neg = -E[log(1 - fD(h̃_j, s_G))]
    ///
    /// where h̃_j are corrupted node embeddings (row-shuffled to break node-graph correspondence).
    /// This pushes negative pairs apart in the representation space.
    /// </summary>
    private Tensor ComputeNegativeLoss(Tensor nodeEmbeddings, Tensor graphRepresentation, Linear discriminator)
    {
        int numNodes = (int)nodeEmbeddings.size(0);

        // Create corrupted view by shuffling node embeddings (breaks node-graph correspondence)
        var indices = torch.randperm(numNodes, device: nodeEmbeddings.device);
        var shuffledNodes = nodeEmbeddings.index_select(0, indices);

        // Expand graph representation
        var expandedGraph = graphRepresentation.expand(numNodes, -1);

        // Concatenate [h̃_j; s_G] for discriminator input
        var pairs = torch.cat(new[] { shuffledNodes, expandedGraph }, dim: 1);

        // Score using discriminator: fD(h̃_j, s_G)
        var scores = discriminator.forward(pairs);
        var negativeScores = functional.sigmoid(scores);

        // BCE loss for negative samples: -log(1 - σ(fD(h̃_j, s_G)))
        var loss = -torch.log(1 - negativeScores + 1e-10f).mean();

        return loss;
    }

    /// <summary>
    /// Samples positive subgraphs based on expertise preference edge weights.
    ///
    /// High-weight edges represent strong expertise preference relationships,
    /// forming semantically coherent subgraphs that capture localized expertise patterns.
    ///
    /// Subgraph sampling helps with:
    /// - Capturing hierarchical structure in expertise preferences
    /// - Learning multi-scale representations (local and global)
    /// - Reducing noise from weak/spurious connections
    /// </summary>
    /// <param name="adjacencyMatrix">Weighted adjacency matrix from expertise preference graph</param>
    /// <param name="threshold">Minimum edge weight to include (default uses instance threshold)</param>
    /// <returns>Subgraph adjacency matrix with only high-weight edges</returns>
    public Tensor SamplePositiveSubgraph(Tensor adjacencyMatrix, float? threshold = null)
    {
        float t = threshold ?? _subgraphThreshold;

        // Select edges with weight above threshold
        var mask = adjacencyMatrix > t;
        return adjacencyMatrix * mask.to_type(ScalarType.Float32);
    }
}

