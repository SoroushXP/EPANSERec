namespace EPANSERec.Core.ReinforcementLearning;

/// <summary>
/// Represents the state in the Markov Decision Process for expertise preference learning.
/// State si is composed of the graph structure information of nodes in the path sequence.
/// </summary>
public class MDPState
{
    /// <summary>
    /// Path sequence of entity IDs visited so far.
    /// </summary>
    public List<int> PathSequence { get; }
    
    /// <summary>
    /// Concatenated feature vectors of nodes in the path [f1; f2; ...; fi].
    /// </summary>
    public float[] StateVector { get; private set; }
    
    /// <summary>
    /// Embedding dimension for each node.
    /// </summary>
    public int EmbeddingDimension { get; }

    public MDPState(int embeddingDimension)
    {
        EmbeddingDimension = embeddingDimension;
        PathSequence = new List<int>();
        StateVector = Array.Empty<float>();
    }

    public MDPState(int embeddingDimension, List<int> pathSequence, float[] stateVector)
    {
        EmbeddingDimension = embeddingDimension;
        PathSequence = new List<int>(pathSequence);
        StateVector = stateVector;
    }

    /// <summary>
    /// Adds a node to the path and updates the state vector.
    /// </summary>
    public void AddNode(int entityId, float[] nodeEmbedding)
    {
        PathSequence.Add(entityId);
        
        // Concatenate the new embedding to the state vector
        var newStateVector = new float[StateVector.Length + nodeEmbedding.Length];
        StateVector.CopyTo(newStateVector, 0);
        nodeEmbedding.CopyTo(newStateVector, StateVector.Length);
        StateVector = newStateVector;
    }

    /// <summary>
    /// Gets the current node (last in path sequence).
    /// </summary>
    public int CurrentNode => PathSequence.Count > 0 ? PathSequence[^1] : -1;

    /// <summary>
    /// Gets the path length.
    /// </summary>
    public int PathLength => PathSequence.Count;

    /// <summary>
    /// Checks if an entity is already in the path (to avoid loops).
    /// </summary>
    public bool ContainsEntity(int entityId) => PathSequence.Contains(entityId);

    /// <summary>
    /// Creates a pooled state representation for Q-network input.
    /// Implements Equation 5-6 from the paper:
    /// O_l = [f_1; f_2; ...; f_(i-1)] - concatenated embeddings of previous nodes
    /// O'_l = sum(O_l) / n - mean pooling over previous embeddings (Equation 6)
    /// s'_i = [O'_l; f_i] - final state representation
    /// Always returns a vector of size EmbeddingDimension * 2.
    /// </summary>
    public float[] GetPooledStateVector()
    {
        var result = new float[EmbeddingDimension * 2];

        if (PathSequence.Count == 0)
        {
            // Return zeros if no path yet
            return result;
        }

        if (PathSequence.Count == 1)
        {
            // Only one node - pooled is zeros, current is the node embedding
            if (StateVector.Length >= EmbeddingDimension)
            {
                Array.Copy(StateVector, 0, result, EmbeddingDimension, EmbeddingDimension);
            }
            return result;
        }

        int numPreviousNodes = PathSequence.Count - 1;
        var pooledVector = new float[EmbeddingDimension];

        // Mean pooling over previous node embeddings (Equation 6: O'_l = sum(O_l) / n)
        for (int d = 0; d < EmbeddingDimension; d++)
        {
            float sum = 0;
            int validCount = 0;
            for (int n = 0; n < numPreviousNodes; n++)
            {
                int idx = n * EmbeddingDimension + d;
                if (idx < StateVector.Length - EmbeddingDimension)
                {
                    sum += StateVector[idx];
                    validCount++;
                }
            }
            pooledVector[d] = validCount > 0 ? sum / validCount : 0;
        }

        // Concatenate pooled vector with last node embedding [O'; fi]
        var lastNodeEmbedding = new float[EmbeddingDimension];
        if (StateVector.Length >= EmbeddingDimension)
        {
            Array.Copy(StateVector, StateVector.Length - EmbeddingDimension,
                       lastNodeEmbedding, 0, EmbeddingDimension);
        }

        pooledVector.CopyTo(result, 0);
        lastNodeEmbedding.CopyTo(result, EmbeddingDimension);

        return result;
    }

    /// <summary>
    /// Creates a copy of this state.
    /// </summary>
    public MDPState Clone()
    {
        return new MDPState(EmbeddingDimension, PathSequence, (float[])StateVector.Clone());
    }
}

