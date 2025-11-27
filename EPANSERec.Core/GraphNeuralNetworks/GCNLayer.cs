using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace EPANSERec.Core.GraphNeuralNetworks;

/// <summary>
/// Graph Convolutional Network layer as described in Equation 10.
/// h_i^l = σ(D̃^(-1/2) * Ã * D̃^(-1/2) * h_i^(l-1) * W)
/// </summary>
public class GCNLayer : Module<Tensor, Tensor, Tensor>
{
    private readonly Linear _linear;
    private readonly bool _useBias;
    private readonly int _inputDim;
    private readonly int _outputDim;

    public GCNLayer(int inputDim, int outputDim, bool useBias = true) 
        : base($"GCNLayer_{inputDim}_{outputDim}")
    {
        _inputDim = inputDim;
        _outputDim = outputDim;
        _useBias = useBias;
        _linear = Linear(inputDim, outputDim, hasBias: useBias);
        
        RegisterComponents();
    }

    /// <summary>
    /// Forward pass for GCN layer.
    /// </summary>
    /// <param name="nodeFeatures">Node feature matrix [N, inputDim]</param>
    /// <param name="normalizedAdjacency">Normalized adjacency matrix D̃^(-1/2) * Ã * D̃^(-1/2) [N, N]</param>
    /// <returns>Output features [N, outputDim]</returns>
    public override Tensor forward(Tensor nodeFeatures, Tensor normalizedAdjacency)
    {
        // Apply adjacency: propagate neighbor information
        var aggregated = torch.matmul(normalizedAdjacency, nodeFeatures);
        
        // Apply linear transformation
        var output = _linear.forward(aggregated);
        
        return output;
    }
}

/// <summary>
/// Multi-layer Graph Convolutional Network for expertise preference optimization.
/// Extended GCN that handles weighted adjacency matrices (expertise preference weight graph).
/// </summary>
public class ExpertiseGCN : Module<Tensor, Tensor, Tensor>
{
    private readonly ModuleList<GCNLayer> _layers;
    private readonly Dropout _dropout;
    private readonly int _numLayers;

    public ExpertiseGCN(int inputDim, int hiddenDim, int outputDim, int numLayers = 2, 
        float dropoutRate = 0.1f) : base("ExpertiseGCN")
    {
        _numLayers = numLayers;
        _layers = new ModuleList<GCNLayer>();
        _dropout = Dropout(dropoutRate);
        
        // First layer
        _layers.Add(new GCNLayer(inputDim, hiddenDim));
        
        // Hidden layers
        for (int i = 1; i < numLayers - 1; i++)
        {
            _layers.Add(new GCNLayer(hiddenDim, hiddenDim));
        }
        
        // Output layer
        if (numLayers > 1)
        {
            _layers.Add(new GCNLayer(hiddenDim, outputDim));
        }
        
        RegisterComponents();
    }

    public override Tensor forward(Tensor nodeFeatures, Tensor adjacencyMatrix)
    {
        // Normalize adjacency matrix with self-loops
        var normalizedAdj = NormalizeAdjacency(adjacencyMatrix);
        
        var x = nodeFeatures;
        
        for (int i = 0; i < _layers.Count; i++)
        {
            x = _layers[i].forward(x, normalizedAdj);
            
            // Apply activation and dropout for all but last layer
            if (i < _layers.Count - 1)
            {
                x = functional.relu(x);
                x = _dropout.forward(x);
            }
        }
        
        return x;
    }

    /// <summary>
    /// Normalizes adjacency matrix: D̃^(-1/2) * Ã * D̃^(-1/2)
    /// where Ã = A + I (adjacency with self-loops)
    /// </summary>
    private Tensor NormalizeAdjacency(Tensor adjacencyMatrix)
    {
        // Add self-loops: Ã = A + I
        var identity = torch.eye(adjacencyMatrix.size(0), device: adjacencyMatrix.device);
        var adjWithSelfLoops = adjacencyMatrix + identity;
        
        // Compute degree matrix D̃
        var degree = adjWithSelfLoops.sum(dim: 1);
        
        // D̃^(-1/2)
        var degreeInvSqrt = torch.pow(degree, -0.5);
        degreeInvSqrt = torch.where(
            torch.isinf(degreeInvSqrt), 
            torch.zeros_like(degreeInvSqrt), 
            degreeInvSqrt);
        
        // D̃^(-1/2) * Ã * D̃^(-1/2)
        var dInvSqrtMat = torch.diag(degreeInvSqrt);
        var normalized = torch.matmul(torch.matmul(dInvSqrtMat, adjWithSelfLoops), dInvSqrtMat);
        
        return normalized;
    }
}

