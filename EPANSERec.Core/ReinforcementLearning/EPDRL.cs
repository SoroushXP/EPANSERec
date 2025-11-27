using EPANSERec.Core.Embeddings;
using EPANSERec.Core.KnowledgeGraph;
using EPANSERec.Core.Models;
using TorchSharp;
using static TorchSharp.torch;

namespace EPANSERec.Core.ReinforcementLearning;

/// <summary>
/// Expertise Preference Deep Reinforcement Learning (EPDRL) module.
/// Uses Double DQN to model historical interactive information of experts
/// and generate expertise preference weight graphs.
/// </summary>
public class EPDRL
{
    private readonly SoftwareKnowledgeGraph _knowledgeGraph;
    private readonly Node2Vec _node2Vec;
    private readonly Dictionary<int, float[]> _nodeEmbeddings;
    private readonly int _embeddingDim;
    private readonly int _maxPathLength;
    private readonly float _rewardPositive;
    private readonly float _rewardNegative;
    private readonly float _gamma; // Discount factor
    private float _epsilon; // Exploration rate (mutable for decay)
    private readonly float _epsilonDecay;
    private readonly float _epsilonMin;
    private readonly int _batchSize;
    private readonly int _memoryCapacity;
    private readonly Random _random;
    
    private QNetwork _policyNetwork;
    private QNetwork _targetNetwork;
    private ReplayMemory _memory;
    private torch.optim.Optimizer _optimizer;

    public EPDRL(
        SoftwareKnowledgeGraph knowledgeGraph,
        int embeddingDim = 100,
        int maxPathLength = 10,
        float rewardPositive = 1.0f,
        float rewardNegative = -1.0f,
        float gamma = 0.99f,
        float epsilon = 1.0f,
        float epsilonDecay = 0.995f,
        float epsilonMin = 0.01f,
        float learningRate = 1e-4f,
        int batchSize = 128,
        int memoryCapacity = 10000,
        int? seed = null)
    {
        _knowledgeGraph = knowledgeGraph;
        _embeddingDim = embeddingDim;
        _maxPathLength = maxPathLength;
        _rewardPositive = rewardPositive;
        _rewardNegative = rewardNegative;
        _gamma = gamma;
        _epsilon = epsilon;
        _epsilonDecay = epsilonDecay;
        _epsilonMin = epsilonMin;
        _batchSize = batchSize;
        _memoryCapacity = memoryCapacity;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
        
        // Train Node2Vec embeddings
        _node2Vec = new Node2Vec(knowledgeGraph, embeddingDim, seed: seed);
        _nodeEmbeddings = _node2Vec.Train();
        
        // Initialize Q-networks
        // Input: pooled state (embeddingDim * 2) + action embedding (embeddingDim)
        int inputDim = embeddingDim * 3;
        _policyNetwork = new QNetwork(inputDim);
        _targetNetwork = new QNetwork(inputDim);
        _targetNetwork.CopyParametersFrom(_policyNetwork);
        
        _memory = new ReplayMemory(memoryCapacity, seed);
        _optimizer = torch.optim.Adam(_policyNetwork.parameters(), lr: learningRate);
    }

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
    /// Runs a single episode of network walking.
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

            if (validNeighbors.Count == 0) break;

            // Select action (epsilon-greedy)
            int actionIdx = SelectAction(state, validNeighbors);
            var nextEntity = validNeighbors[actionIdx].neighbor;

            // Calculate reward
            float reward;
            bool done = false;

            if (targetEntities.Contains(nextEntity.Id))
            {
                reward = _rewardPositive;
                done = true;
            }
            else if (state.PathLength >= _maxPathLength - 1)
            {
                reward = _rewardNegative;
                done = true;
            }
            else
            {
                reward = -0.01f; // Small negative reward to encourage shorter paths
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
            _memory.Push(new Experience(stateInput, actionIdx, reward, nextStateInput, done));

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
    /// Trains the Q-network on a batch of experiences.
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

        // Compute target Q-values (no gradient needed for target)
        Tensor targetQ;
        using (torch.no_grad())
        {
            var nextStates = batch.Select(e => e.NextState).ToArray();
            using var nextStatesTensor = torch.tensor(nextStates.SelectMany(s => s).ToArray())
                .reshape(_batchSize, -1);
            using var nextQ = _targetNetwork.forward(nextStatesTensor);
            targetQ = rewardsTensor + _gamma * nextQ * donesTensor;
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
