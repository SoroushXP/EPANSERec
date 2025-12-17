using EPANSERec.Core.Embeddings;
using EPANSERec.Core.KnowledgeGraph;
using EPANSERec.Core.Models;
using TorchSharp;
using static TorchSharp.torch;

namespace EPANSERec.Core.ReinforcementLearning;

/// <summary>
/// Expertise Preference Deep Reinforcement Learning (EPDRL) module.
///
/// Implements Section 4.1 of the EPAN-SERec paper using Double DQN to model
/// historical interactive information of experts and generate expertise preference weight graphs.
///
/// ═══════════════════════════════════════════════════════════════════════════════
/// MDP FORMULATION (Section 4.1)
/// ═══════════════════════════════════════════════════════════════════════════════
///
/// The MDP M = (S, A, O, R) is defined as:
///
/// **State Space S:**
///   s_i = [O'_l; f_i] where:
///   - O'_l = mean-pooled embeddings of previously visited nodes (Eq. 5-6)
///   - f_i = embedding of current node
///
/// **Action Space A:**
///   A(s_i) = {neighbors of current node not yet visited}
///   Actions are represented by node embeddings
///
/// **Transition O:**
///   Deterministic transitions in the knowledge graph
///   s_{i+1} = AddNode(s_i, a_i) where a_i is the selected neighbor
///
/// **Reward Function R:**
///   Based on the paper's conceptual description, rewards encourage:
///   1. Reaching target entities (expert's historical interactions)
///   2. Shorter paths (efficient traversal)
///   3. Avoiding dead-ends
///
/// ═══════════════════════════════════════════════════════════════════════════════
/// REWARD FUNCTION DESIGN
/// ═══════════════════════════════════════════════════════════════════════════════
///
/// The paper defines rewards conceptually as positive for reaching target entities
/// and negative otherwise. Our implementation uses:
///
///   R(s, a, s') =
///     +R_pos  if s' is a target entity (expert's historical entity)     → Success
///     -R_neg  if max path length reached without success                → Failure
///     -R_step otherwise (small step penalty)                            → Ongoing
///
/// where:
///   - R_pos = 1.0 (configurable): Reward for discovering expertise preference path
///   - R_neg = 1.0 (configurable): Penalty for unsuccessful episode
///   - R_step = 0.01: Small step cost to encourage shorter paths
///
/// This reward shaping encourages:
///   1. Finding connections between expert's historical entities (maximizes expertise coverage)
///   2. Preferring shorter paths (more direct expertise relationships)
///   3. Exploring diverse paths (through ε-greedy exploration)
///
/// ═══════════════════════════════════════════════════════════════════════════════
/// DOUBLE DQN TRAINING
/// ═══════════════════════════════════════════════════════════════════════════════
///
/// Uses Double DQN (van Hasselt et al., 2016) to reduce Q-value overestimation:
///   Q_target = r + γ * Q_target(s', argmax_a Q_policy(s', a))
///
/// Key components:
/// - Policy network: Selects best action via argmax
/// - Target network: Evaluates the selected action
/// - Experience replay: Stabilizes training with random sampling
/// - ε-decay: Balances exploration vs exploitation
/// </summary>
public class EPDRL
{
    private readonly SoftwareKnowledgeGraph _knowledgeGraph;
    private readonly Node2Vec? _node2Vec;
    private readonly Dictionary<int, float[]> _nodeEmbeddings;
    private readonly int _embeddingDim;
    private readonly int _maxPathLength;
    private readonly float _rewardPositive;    // R_pos: Reward for reaching target entity
    private readonly float _rewardNegative;    // R_neg: Penalty for episode failure
    private readonly float _rewardStep;        // R_step: Small per-step cost for path efficiency
    private readonly float _gamma;             // Discount factor (typical: 0.99)
    private float _epsilon;                    // Exploration rate (decays during training)
    private readonly float _epsilonDecay;      // Decay rate per episode
    private readonly float _epsilonMin;        // Minimum exploration rate
    private readonly int _batchSize;
    private readonly int _memoryCapacity;
    private readonly Random _random;

    private QNetwork _policyNetwork;
    private QNetwork _targetNetwork;
    private ReplayMemory _memory;
    private torch.optim.Optimizer _optimizer;

    /// <summary>
    /// Initializes the EPDRL module with configurable hyperparameters.
    /// </summary>
    /// <param name="knowledgeGraph">Software knowledge graph for path finding</param>
    /// <param name="embeddingDim">Node embedding dimension (default: 100)</param>
    /// <param name="maxPathLength">Maximum path length before episode terminates (default: 10)</param>
    /// <param name="rewardPositive">Reward for reaching target entity (default: 1.0)</param>
    /// <param name="rewardNegative">Penalty for episode failure (default: 1.0)</param>
    /// <param name="rewardStep">Per-step cost to encourage shorter paths (default: 0.01)</param>
    /// <param name="gamma">Discount factor for future rewards (default: 0.99, standard DRL)</param>
    /// <param name="epsilon">Initial exploration rate (default: 1.0, full exploration)</param>
    /// <param name="epsilonDecay">Exploration decay per episode (default: 0.995)</param>
    /// <param name="epsilonMin">Minimum exploration rate (default: 0.01)</param>
    /// <param name="learningRate">Adam optimizer learning rate (default: 1e-4)</param>
    /// <param name="batchSize">Experience replay batch size (default: 128)</param>
    /// <param name="memoryCapacity">Replay buffer capacity (default: 10000)</param>
    /// <param name="seed">Random seed for reproducibility</param>
    /// <param name="pretrainedEmbeddings">Optional pre-trained node embeddings to avoid retraining Node2Vec</param>
    public EPDRL(
        SoftwareKnowledgeGraph knowledgeGraph,
        int embeddingDim = 100,
        int maxPathLength = 10,
        float rewardPositive = 1.0f,
        float rewardNegative = 1.0f,
        float rewardStep = 0.01f,
        float gamma = 0.99f,
        float epsilon = 1.0f,
        float epsilonDecay = 0.995f,
        float epsilonMin = 0.01f,
        float learningRate = 1e-4f,
        int batchSize = 128,
        int memoryCapacity = 10000,
        int? seed = null,
        Dictionary<int, float[]>? pretrainedEmbeddings = null)
    {
        _knowledgeGraph = knowledgeGraph;
        _embeddingDim = embeddingDim;
        _maxPathLength = maxPathLength;
        _rewardPositive = rewardPositive;
        _rewardNegative = rewardNegative;
        _rewardStep = rewardStep;
        _gamma = gamma;
        _epsilon = epsilon;
        _epsilonDecay = epsilonDecay;
        _epsilonMin = epsilonMin;
        _batchSize = batchSize;
        _memoryCapacity = memoryCapacity;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();

        // Use pre-trained embeddings if provided, otherwise train Node2Vec
        if (pretrainedEmbeddings != null)
        {
            _node2Vec = null;
            _nodeEmbeddings = pretrainedEmbeddings;
        }
        else
        {
            _node2Vec = new Node2Vec(knowledgeGraph, embeddingDim, seed: seed);
            _nodeEmbeddings = _node2Vec.Train();
        }

        // Initialize Q-networks for Double DQN
        // Input: pooled state (embeddingDim * 2) + action embedding (embeddingDim) = embeddingDim * 3
        int inputDim = embeddingDim * 3;
        _policyNetwork = new QNetwork(inputDim);
        _targetNetwork = new QNetwork(inputDim);
        _targetNetwork.CopyParametersFrom(_policyNetwork);

        _memory = new ReplayMemory(memoryCapacity, seed);
        _optimizer = torch.optim.Adam(_policyNetwork.parameters(), lr: learningRate);
    }

    /// <summary>
    /// Gets the node embeddings (useful for sharing with other EPDRL instances).
    /// </summary>
    public Dictionary<int, float[]> NodeEmbeddings => _nodeEmbeddings;

    /// <summary>
    /// Generates the expertise preference weight graph for an expert.
    /// Implements Algorithm 1 from the paper.
    /// </summary>
    public ExpertisePreferenceWeightGraph GeneratePreferenceGraph(Expert expert, int episodes = 100)
    {
        var preferenceGraph = new ExpertisePreferenceWeightGraph(expert.Id);
        var pathRewards = new List<(List<int> path, float reward)>();
        
        for (int episode = 0; episode < episodes; episode++)
        {
            // Decay epsilon
            _epsilon = Math.Max(_epsilonMin, _epsilon * _epsilonDecay);
            
            foreach (var startEntityId in expert.HistoricalEntityIds)
            {
                var (path, reward) = RunEpisode(startEntityId, expert.HistoricalEntityIds);
                if (path.Count > 1)
                {
                    pathRewards.Add((path, reward));
                }
            }
            
            // Train on batch if enough experiences
            if (_memory.CanSample(_batchSize))
            {
                TrainOnBatch();
            }
            
            // Update target network periodically
            if (episode % 10 == 0)
            {
                _targetNetwork.CopyParametersFrom(_policyNetwork);
            }
        }
        
        // Calculate edge weights from paths (Equation 3)
        foreach (var (path, reward) in pathRewards)
        {
            AssignPathWeights(preferenceGraph, path, reward);
        }
        
        preferenceGraph.NormalizeWeights();
        preferenceGraph.CalculateSelfLoopWeights();

        return preferenceGraph;
    }

    /// <summary>
    /// Runs a single episode of network walking (trajectory generation).
    ///
    /// Episode terminates when:
    /// 1. Target entity is reached (success, +R_pos reward)
    /// 2. Max path length is reached (failure, -R_neg penalty)
    /// 3. No valid neighbors remain (dead end, -R_neg penalty)
    ///
    /// At each step, -R_step penalty encourages finding shorter paths.
    /// </summary>
    private (List<int> path, float reward) RunEpisode(int startEntityId, HashSet<int> targetEntities)
    {
        var state = new MDPState(_embeddingDim);
        var startEmbedding = GetNodeEmbedding(startEntityId);
        state.AddNode(startEntityId, startEmbedding);

        float totalReward = 0;

        while (state.PathLength < _maxPathLength)
        {
            var neighbors = _knowledgeGraph.GetNeighbors(state.CurrentNode);
            var validNeighbors = neighbors
                .Where(n => !state.ContainsEntity(n.neighbor.Id))
                .ToList();

            // Dead end: no valid neighbors to explore
            if (validNeighbors.Count == 0) break;

            // Select action using ε-greedy policy
            int actionIdx = SelectAction(state, validNeighbors);
            var nextEntity = validNeighbors[actionIdx].neighbor;

            // ═══════════════════════════════════════════════════════════════════
            // REWARD FUNCTION (as described in paper Section 4.1)
            // ═══════════════════════════════════════════════════════════════════
            // R(s, a, s') =
            //   +R_pos  if s' is a target entity → Successfully found expertise path
            //   -R_neg  if max path length reached → Episode failure
            //   -R_step otherwise → Step penalty for path efficiency
            // ═══════════════════════════════════════════════════════════════════
            float reward;
            bool done = false;

            if (targetEntities.Contains(nextEntity.Id))
            {
                // SUCCESS: Found a target entity (expert's historical interaction)
                // Positive reward encourages finding paths connecting expertise areas
                reward = _rewardPositive;
                done = true;
            }
            else if (state.PathLength >= _maxPathLength - 1)
            {
                // FAILURE: Max path length reached without finding target
                // Negative reward discourages long, unsuccessful paths
                reward = -_rewardNegative;
                done = true;
            }
            else
            {
                // ONGOING: Small step penalty to encourage shorter paths
                // This implements implicit path length regularization from the paper
                reward = -_rewardStep;
            }

            totalReward += reward;

            // Create next state
            var nextState = state.Clone();
            nextState.AddNode(nextEntity.Id, GetNodeEmbedding(nextEntity.Id));

            // Store experience - both states use the same format (pooled state + action)
            var actionEmb = GetNodeEmbedding(nextEntity.Id);
            var stateInput = CreateNetworkInput(state, actionEmb);
            // For next state, we use a zero action embedding as placeholder
            var nextStateInput = CreateNetworkInput(nextState, new float[_embeddingDim]);

            // For Double DQN: store available actions at next state
            List<float[]>? nextStateActionInputs = null;
            if (!done)
            {
                var nextNeighbors = _knowledgeGraph.GetNeighbors(nextState.CurrentNode);
                var validNextNeighbors = nextNeighbors
                    .Where(n => !nextState.ContainsEntity(n.neighbor.Id))
                    .ToList();

                if (validNextNeighbors.Count > 0)
                {
                    nextStateActionInputs = validNextNeighbors
                        .Select(n => CreateNetworkInput(nextState, GetNodeEmbedding(n.neighbor.Id)))
                        .ToList();
                }
            }

            _memory.Push(new Experience(stateInput, actionIdx, reward, nextStateInput, done, nextStateActionInputs));

            state = nextState;

            if (done) break;
        }

        return (state.PathSequence, totalReward);
    }

    /// <summary>
    /// Selects an action using epsilon-greedy policy.
    /// </summary>
    private int SelectAction(MDPState state, List<(Entity neighbor, Relation relation, float weight)> actions)
    {
        if (_random.NextDouble() < _epsilon)
        {
            return _random.Next(actions.Count);
        }

        // Evaluate Q-values for all actions
        float bestQ = float.MinValue;
        int bestAction = 0;

        for (int i = 0; i < actions.Count; i++)
        {
            var actionEmb = GetNodeEmbedding(actions[i].neighbor.Id);
            var input = CreateNetworkInput(state, actionEmb);

            using var inputTensor = torch.tensor(input).unsqueeze(0);
            using var qValue = _policyNetwork.forward(inputTensor);
            float q = qValue.item<float>();

            if (q > bestQ)
            {
                bestQ = q;
                bestAction = i;
            }
        }

        return bestAction;
    }

    /// <summary>
    /// Creates network input from state and action.
    /// </summary>
    private float[] CreateNetworkInput(MDPState state, float[] actionEmbedding)
    {
        var pooledState = state.GetPooledStateVector();
        var input = new float[pooledState.Length + actionEmbedding.Length];
        pooledState.CopyTo(input, 0);
        actionEmbedding.CopyTo(input, pooledState.Length);
        return input;
    }

    /// <summary>
    /// Trains the Q-network on a batch of experiences using Double DQN.
    /// Double DQN: Use policy network to select best action, target network to evaluate it.
    /// Q_target = r + γ * Q_target(s', argmax_a Q_policy(s', a))
    /// This reduces overestimation bias compared to standard DQN.
    /// </summary>
    private void TrainOnBatch()
    {
        var batch = _memory.Sample(_batchSize);

        var states = batch.Select(e => e.State).ToArray();
        var rewards = batch.Select(e => e.Reward).ToArray();
        var dones = batch.Select(e => e.Done ? 0f : 1f).ToArray();

        using var statesTensor = torch.tensor(states.SelectMany(s => s).ToArray(), requires_grad: false)
            .reshape(_batchSize, -1);
        using var rewardsTensor = torch.tensor(rewards).unsqueeze(1);
        using var donesTensor = torch.tensor(dones).unsqueeze(1);

        // Compute target Q-values using Double DQN (no gradient needed)
        Tensor targetQ;
        using (torch.no_grad())
        {
            var nextQValues = new float[_batchSize];

            for (int i = 0; i < batch.Count; i++)
            {
                var experience = batch[i];

                if (experience.Done || experience.NextStateActionInputs == null ||
                    experience.NextStateActionInputs.Count == 0)
                {
                    // Terminal state or no available actions: Q-value is 0
                    nextQValues[i] = 0f;
                }
                else
                {
                    // Double DQN:
                    // Step 1: Use POLICY network to select best action (argmax)
                    int bestActionIdx = 0;
                    float bestPolicyQ = float.MinValue;

                    for (int a = 0; a < experience.NextStateActionInputs.Count; a++)
                    {
                        using var actionInput = torch.tensor(experience.NextStateActionInputs[a]).unsqueeze(0);
                        using var qValue = _policyNetwork.forward(actionInput);
                        float q = qValue.item<float>();

                        if (q > bestPolicyQ)
                        {
                            bestPolicyQ = q;
                            bestActionIdx = a;
                        }
                    }

                    // Step 2: Use TARGET network to evaluate that action
                    using var bestActionInput = torch.tensor(experience.NextStateActionInputs[bestActionIdx]).unsqueeze(0);
                    using var targetQValue = _targetNetwork.forward(bestActionInput);
                    nextQValues[i] = targetQValue.item<float>();
                }
            }

            using var nextQTensor = torch.tensor(nextQValues).unsqueeze(1);
            targetQ = rewardsTensor + _gamma * nextQTensor * donesTensor;
        }

        // Current Q-values (gradient needed)
        _optimizer.zero_grad();
        using var currentQ = _policyNetwork.forward(statesTensor);

        // Compute loss and update
        using var loss = torch.nn.functional.mse_loss(currentQ, targetQ.detach());
        loss.backward();
        _optimizer.step();

        targetQ.Dispose();
    }

    /// <summary>
    /// Assigns weights to edges in a path based on reward (Equation 3).
    /// w(eij) = (1/e^d) * Ri, where d is distance to target node.
    /// </summary>
    private void AssignPathWeights(ExpertisePreferenceWeightGraph graph, List<int> path, float reward)
    {
        if (reward <= 0 || path.Count < 2) return;

        for (int i = 0; i < path.Count - 1; i++)
        {
            int distance = path.Count - 1 - i; // Distance to target (last node)
            float weight = (1.0f / (float)Math.Exp(distance)) * reward;

            float existingWeight = graph.GetEdgeWeight(path[i], path[i + 1]);
            graph.SetEdgeWeight(path[i], path[i + 1], existingWeight + weight);
        }
    }

    /// <summary>
    /// Gets node embedding, returns zero vector if not found.
    /// </summary>
    private float[] GetNodeEmbedding(int nodeId)
    {
        return _nodeEmbeddings.TryGetValue(nodeId, out var emb)
            ? emb
            : new float[_embeddingDim];
    }
}
