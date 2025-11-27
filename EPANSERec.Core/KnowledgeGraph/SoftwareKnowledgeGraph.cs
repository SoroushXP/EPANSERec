namespace EPANSERec.Core.KnowledgeGraph;

/// <summary>
/// Represents the Software Knowledge Graph (SWKG) as described in the paper.
/// Contains entities, relations, and triples with adjacency information.
/// </summary>
public class SoftwareKnowledgeGraph
{
    private readonly Dictionary<int, Entity> _entities = new();
    private readonly Dictionary<int, Relation> _relations = new();
    private readonly List<Triple> _triples = new();
    private readonly Dictionary<int, List<(int neighborId, int relationId, float weight)>> _adjacencyList = new();
    
    public IReadOnlyDictionary<int, Entity> Entities => _entities;
    public IReadOnlyDictionary<int, Relation> Relations => _relations;
    public IReadOnlyList<Triple> Triples => _triples;
    
    public int EntityCount => _entities.Count;
    public int RelationCount => _relations.Count;
    public int TripleCount => _triples.Count;

    /// <summary>
    /// Adds an entity to the knowledge graph.
    /// </summary>
    public void AddEntity(Entity entity)
    {
        _entities[entity.Id] = entity;
        if (!_adjacencyList.ContainsKey(entity.Id))
            _adjacencyList[entity.Id] = new List<(int, int, float)>();
    }

    /// <summary>
    /// Adds a relation type to the knowledge graph.
    /// </summary>
    public void AddRelation(Relation relation)
    {
        _relations[relation.Id] = relation;
    }

    /// <summary>
    /// Adds a triple to the knowledge graph and updates adjacency list.
    /// </summary>
    public void AddTriple(Triple triple, float weight = 1.0f)
    {
        _triples.Add(triple);

        // Add bidirectional edges for undirected graph traversal
        if (!_adjacencyList.ContainsKey(triple.Head.Id))
            _adjacencyList[triple.Head.Id] = new List<(int, int, float)>();
        if (!_adjacencyList.ContainsKey(triple.Tail.Id))
            _adjacencyList[triple.Tail.Id] = new List<(int, int, float)>();

        _adjacencyList[triple.Head.Id].Add((triple.Tail.Id, triple.Relation.Id, weight));
        _adjacencyList[triple.Tail.Id].Add((triple.Head.Id, triple.Relation.Id, weight));
    }

    /// <summary>
    /// Adds a triple by entity and relation IDs.
    /// </summary>
    public void AddTriple(int headId, int relationId, int tailId, float weight = 1.0f)
    {
        if (_entities.TryGetValue(headId, out var head) &&
            _entities.TryGetValue(tailId, out var tail) &&
            _relations.TryGetValue(relationId, out var relation))
        {
            var triple = new Triple(head, relation, tail);
            AddTriple(triple, weight);
        }
        else
        {
            // If entities/relations don't exist, just add to adjacency list
            if (!_adjacencyList.ContainsKey(headId))
                _adjacencyList[headId] = new List<(int, int, float)>();
            if (!_adjacencyList.ContainsKey(tailId))
                _adjacencyList[tailId] = new List<(int, int, float)>();

            _adjacencyList[headId].Add((tailId, relationId, weight));
            _adjacencyList[tailId].Add((headId, relationId, weight));
        }
    }

    /// <summary>
    /// Gets neighbors of an entity.
    /// </summary>
    public List<(Entity neighbor, Relation relation, float weight)> GetNeighbors(int entityId)
    {
        var result = new List<(Entity, Relation, float)>();
        if (_adjacencyList.TryGetValue(entityId, out var neighbors))
        {
            foreach (var (neighborId, relationId, weight) in neighbors)
            {
                if (_entities.TryGetValue(neighborId, out var neighbor) &&
                    _relations.TryGetValue(relationId, out var relation))
                {
                    result.Add((neighbor, relation, weight));
                }
            }
        }
        return result;
    }

    /// <summary>
    /// Gets the entity by ID.
    /// </summary>
    public Entity? GetEntity(int id) => _entities.GetValueOrDefault(id);

    /// <summary>
    /// Gets the relation by ID.
    /// </summary>
    public Relation? GetRelation(int id) => _relations.GetValueOrDefault(id);

    /// <summary>
    /// Gets the degree of an entity (number of connections).
    /// </summary>
    public int GetDegree(int entityId) => 
        _adjacencyList.TryGetValue(entityId, out var neighbors) ? neighbors.Count : 0;

    /// <summary>
    /// Updates edge weight between two entities.
    /// </summary>
    public void UpdateEdgeWeight(int entityId1, int entityId2, float newWeight)
    {
        UpdateWeightInList(entityId1, entityId2, newWeight);
        UpdateWeightInList(entityId2, entityId1, newWeight);
    }

    private void UpdateWeightInList(int fromId, int toId, float newWeight)
    {
        if (_adjacencyList.TryGetValue(fromId, out var neighbors))
        {
            for (int i = 0; i < neighbors.Count; i++)
            {
                if (neighbors[i].neighborId == toId)
                {
                    neighbors[i] = (toId, neighbors[i].relationId, newWeight);
                    break;
                }
            }
        }
    }

    /// <summary>
    /// Gets the adjacency matrix as a 2D float array.
    /// </summary>
    public float[,] GetAdjacencyMatrix()
    {
        int n = _entities.Count;
        var matrix = new float[n, n];
        var idToIndex = _entities.Keys.Select((id, idx) => (id, idx)).ToDictionary(x => x.id, x => x.idx);
        
        foreach (var (entityId, neighbors) in _adjacencyList)
        {
            if (idToIndex.TryGetValue(entityId, out var i))
            {
                foreach (var (neighborId, _, weight) in neighbors)
                {
                    if (idToIndex.TryGetValue(neighborId, out var j))
                        matrix[i, j] = weight;
                }
            }
        }
        return matrix;
    }
}

