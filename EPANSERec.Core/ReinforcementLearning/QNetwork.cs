using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace EPANSERec.Core.ReinforcementLearning;

/// <summary>
/// Deep Q-Network for expertise preference learning (EPDRL component).
///
/// Used in Double DQN setup from the EPAN-SERec paper (Section 4.1) where:
/// - Policy network: Selects best action (argmax_a Q_policy(s', a))
/// - Target network: Evaluates selected action (Q_target(s', a*))
///
/// This separation reduces overestimation bias in Q-learning.
///
/// ═══════════════════════════════════════════════════════════════════════════════
/// ARCHITECTURE DESIGN RATIONALE
/// ═══════════════════════════════════════════════════════════════════════════════
///
/// Input representation (Equations 5-6 in paper):
///   [O'_l; f_i; a_i] where:
///   - O'_l = mean-pooled embeddings of previous nodes in path (history context)
///   - f_i  = embedding of current node (current state)
///   - a_i  = embedding of candidate action node (action representation)
///
///   Total input dim = embeddingDim * 3 (e.g., 100*3 = 300 for default)
///
/// Network architecture:
///   Layer 1: input_dim → 256 (expansion to capture complex patterns)
///   Layer 2: 256 → 256 (same dimensionality for residual-like processing)
///   Layer 3: 256 → 128 (gradual reduction toward output)
///   Layer 4: 128 → 1 (single Q-value output)
///
/// Design choices aligned with DRL best practices:
/// - 4 layers: Deep enough for complex state-action mappings, not too deep to cause
///   training instability (Mnih et al., 2015 - DQN paper uses similar depth)
/// - Hidden dim 256: Standard choice balancing capacity and computation
/// - ReLU activation: Standard for DQN, avoids vanishing gradients
/// - Dropout (0.1): Light regularization to prevent overfitting on replay buffer
/// - No batch normalization: Can interfere with target network stability in DQN
///
/// The paper does not specify exact architecture, so this follows standard DQN
/// conventions from Mnih et al. (2015) "Human-level control through deep RL"
/// ═══════════════════════════════════════════════════════════════════════════════
/// </summary>
public class QNetwork : Module<Tensor, Tensor>
{
    private readonly Linear _fc1;
    private readonly Linear _fc2;
    private readonly Linear _fc3;
    private readonly Linear _fc4;
    private readonly Dropout _dropout;

    /// <summary>
    /// Creates a Q-Network for state-action value estimation.
    /// </summary>
    /// <param name="inputDim">Input dimension: embeddingDim * 3 = [pooled_history; current_node; action]</param>
    /// <param name="hiddenDim">Hidden layer dimension (default: 256, standard DQN choice)</param>
    /// <param name="dropoutRate">Dropout rate for regularization (default: 0.1, light regularization)</param>
    public QNetwork(int inputDim, int hiddenDim = 256, float dropoutRate = 0.1f)
        : base("QNetwork")
    {
        // Layer 1: Expand input to hidden dimension
        // Captures interactions between state components (history, current, action)
        _fc1 = Linear(inputDim, hiddenDim);

        // Layer 2: Same dimension for deeper feature extraction
        // Allows complex non-linear transformations without information bottleneck
        _fc2 = Linear(hiddenDim, hiddenDim);

        // Layer 3: Gradual reduction toward output
        // Compresses features into more abstract representation
        _fc3 = Linear(hiddenDim, hiddenDim / 2);

        // Layer 4: Output single Q-value
        // Q(s, a) estimates expected cumulative reward from state s taking action a
        _fc4 = Linear(hiddenDim / 2, 1);

        // Light dropout for regularization (too much hurts DQN stability)
        _dropout = Dropout(dropoutRate);

        RegisterComponents();
    }

    /// <summary>
    /// Forward pass computing Q(s, a) value.
    ///
    /// Architecture: FC(ReLU) → Dropout → FC(ReLU) → Dropout → FC(ReLU) → FC
    /// No activation on output (Q-values can be any real number).
    /// </summary>
    public override Tensor forward(Tensor input)
    {
        var x = functional.relu(_fc1.forward(input));
        x = _dropout.forward(x);
        x = functional.relu(_fc2.forward(x));
        x = _dropout.forward(x);
        x = functional.relu(_fc3.forward(x));
        // No activation on output: Q-values are unbounded
        return _fc4.forward(x);
    }

    /// <summary>
    /// Copies parameters from source network (for target network updates).
    ///
    /// In Double DQN, target network is updated periodically (every N episodes)
    /// rather than every step, providing stable Q-value targets during training.
    /// </summary>
    public void CopyParametersFrom(QNetwork source)
    {
        this.load_state_dict(source.state_dict());
    }
}

/// <summary>
/// Dueling DQN architecture for improved value estimation.
///
/// Based on Wang et al. (2016) "Dueling Network Architectures for Deep Reinforcement Learning"
///
/// The key insight is to decompose Q(s, a) into:
///   Q(s, a) = V(s) + A(s, a) - mean(A(s, a'))
///
/// where:
/// - V(s) = state value function (how good is this state regardless of action)
/// - A(s, a) = advantage function (how much better is action a than average)
///
/// Benefits over standard DQN:
/// 1. Better generalization: V(s) can be learned even when many actions have similar values
/// 2. More stable learning: Separating value from advantage reduces variance
/// 3. Better for sparse rewards: State value can be learned without exploring all actions
///
/// Note: This is an alternative architecture option not used by default in EPDRL,
/// but available for experimentation. The paper uses standard Double DQN.
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

        // Shared feature layer: extracts common state features
        _featureLayer = Linear(inputDim, hiddenDim);

        // Value stream: V(s) - estimates state value independent of action
        _valueStream1 = Linear(hiddenDim, hiddenDim / 2);
        _valueStream2 = Linear(hiddenDim / 2, 1);

        // Advantage stream: A(s, a) - estimates relative advantage of each action
        _advantageStream1 = Linear(hiddenDim, hiddenDim / 2);
        _advantageStream2 = Linear(hiddenDim / 2, actionSize);

        _dropout = Dropout(dropoutRate);

        RegisterComponents();
    }

    /// <summary>
    /// Forward pass computing Q(s,a) using dueling architecture.
    ///
    /// Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
    ///
    /// The mean subtraction ensures identifiability: without it, V and A could
    /// be shifted by a constant without changing Q, making learning unstable.
    /// </summary>
    public override Tensor forward(Tensor input)
    {
        // Shared feature extraction
        var features = functional.relu(_featureLayer.forward(input));
        features = _dropout.forward(features);

        // Value stream: V(s)
        var value = functional.relu(_valueStream1.forward(features));
        value = _valueStream2.forward(value);

        // Advantage stream: A(s, a) for all actions
        var advantage = functional.relu(_advantageStream1.forward(features));
        advantage = _advantageStream2.forward(advantage);

        // Combine using mean centering for identifiability:
        // Q(s,a) = V(s) + (A(s,a) - mean_a'(A(s,a')))
        var advantageMean = advantage.mean(dimensions: new long[] { -1 }, keepdim: true);
        var qValues = value + advantage - advantageMean;

        return qValues;
    }

    /// <summary>
    /// Copies parameters for target network updates.
    /// </summary>
    public void CopyParametersFrom(DuelingQNetwork source)
    {
        this.load_state_dict(source.state_dict());
    }
}

