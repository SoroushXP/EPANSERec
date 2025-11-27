using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace EPANSERec.Core.ReinforcementLearning;

/// <summary>
/// Deep Q-Network for expertise preference learning.
/// Input: [pooled state vector O'; current node fi; action ai]
/// Output: Q-value for the state-action pair.
/// </summary>
public class QNetwork : Module<Tensor, Tensor>
{
    private readonly Linear _fc1;
    private readonly Linear _fc2;
    private readonly Linear _fc3;
    private readonly Linear _fc4;
    private readonly Dropout _dropout;

    public QNetwork(int inputDim, int hiddenDim = 256, float dropoutRate = 0.1f) 
        : base("QNetwork")
    {
        _fc1 = Linear(inputDim, hiddenDim);
        _fc2 = Linear(hiddenDim, hiddenDim);
        _fc3 = Linear(hiddenDim, hiddenDim / 2);
        _fc4 = Linear(hiddenDim / 2, 1);
        _dropout = Dropout(dropoutRate);
        
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = functional.relu(_fc1.forward(input));
        x = _dropout.forward(x);
        x = functional.relu(_fc2.forward(x));
        x = _dropout.forward(x);
        x = functional.relu(_fc3.forward(x));
        return _fc4.forward(x);
    }

    /// <summary>
    /// Creates a copy of network parameters for target network.
    /// </summary>
    public void CopyParametersFrom(QNetwork source)
    {
        this.load_state_dict(source.state_dict());
    }
}

/// <summary>
/// Dueling DQN architecture for better value estimation.
/// Separates value and advantage streams.
/// </summary>
public class DuelingQNetwork : Module<Tensor, Tensor>
{
    private readonly Linear _featureLayer;
    private readonly Linear _valueStream1;
    private readonly Linear _valueStream2;
    private readonly Linear _advantageStream1;
    private readonly Linear _advantageStream2;
    private readonly Dropout _dropout;
    private readonly int _actionSize;

    public DuelingQNetwork(int inputDim, int actionSize, int hiddenDim = 256, float dropoutRate = 0.1f)
        : base("DuelingQNetwork")
    {
        _actionSize = actionSize;
        
        // Shared feature layer
        _featureLayer = Linear(inputDim, hiddenDim);
        
        // Value stream: V(s)
        _valueStream1 = Linear(hiddenDim, hiddenDim / 2);
        _valueStream2 = Linear(hiddenDim / 2, 1);
        
        // Advantage stream: A(s, a)
        _advantageStream1 = Linear(hiddenDim, hiddenDim / 2);
        _advantageStream2 = Linear(hiddenDim / 2, actionSize);
        
        _dropout = Dropout(dropoutRate);
        
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // Shared features
        var features = functional.relu(_featureLayer.forward(input));
        features = _dropout.forward(features);
        
        // Value stream
        var value = functional.relu(_valueStream1.forward(features));
        value = _valueStream2.forward(value);
        
        // Advantage stream
        var advantage = functional.relu(_advantageStream1.forward(features));
        advantage = _advantageStream2.forward(advantage);
        
        // Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        var advantageMean = advantage.mean(dimensions: new long[] { -1 }, keepdim: true);
        var qValues = value + advantage - advantageMean;
        
        return qValues;
    }

    /// <summary>
    /// Creates a copy of network parameters for target network.
    /// </summary>
    public void CopyParametersFrom(DuelingQNetwork source)
    {
        this.load_state_dict(source.state_dict());
    }
}

