namespace EPANSERec.Core.ReinforcementLearning;

/// <summary>
/// Represents a single experience tuple for replay memory.
/// </summary>
public class Experience
{
    public float[] State { get; set; }
    public int Action { get; set; }
    public float Reward { get; set; }
    public float[] NextState { get; set; }
    public bool Done { get; set; }

    public Experience(float[] state, int action, float reward, float[] nextState, bool done)
    {
        State = state;
        Action = action;
        Reward = reward;
        NextState = nextState;
        Done = done;
    }
}

/// <summary>
/// Replay memory buffer for experience replay in DQN.
/// </summary>
public class ReplayMemory
{
    private readonly List<Experience> _memory;
    private readonly int _capacity;
    private readonly Random _random;
    private int _position;

    public ReplayMemory(int capacity, int? seed = null)
    {
        _capacity = capacity;
        _memory = new List<Experience>(capacity);
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
        _position = 0;
    }

    /// <summary>
    /// Adds an experience to the replay memory.
    /// </summary>
    public void Push(Experience experience)
    {
        if (_memory.Count < _capacity)
        {
            _memory.Add(experience);
        }
        else
        {
            _memory[_position] = experience;
        }
        _position = (_position + 1) % _capacity;
    }

    /// <summary>
    /// Samples a random batch from the replay memory.
    /// </summary>
    public List<Experience> Sample(int batchSize)
    {
        var batch = new List<Experience>();
        var indices = Enumerable.Range(0, _memory.Count)
            .OrderBy(_ => _random.Next())
            .Take(Math.Min(batchSize, _memory.Count))
            .ToList();
            
        foreach (var idx in indices)
        {
            batch.Add(_memory[idx]);
        }
        return batch;
    }

    public int Count => _memory.Count;
    public bool CanSample(int batchSize) => _memory.Count >= batchSize;
}

