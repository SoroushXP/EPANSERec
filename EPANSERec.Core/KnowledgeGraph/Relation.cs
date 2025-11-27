namespace EPANSERec.Core.KnowledgeGraph;

/// <summary>
/// Represents a relation type in the software knowledge graph.
/// Based on SWKG: 5 relation types defined in the paper.
/// </summary>
public enum RelationType
{
    UsedBy,
    DependsOn,
    ImplementedIn,
    RelatedTo,
    PartOf,
    // Additional types for flexibility
    Uses,
    BelongsTo,
    HasTag,
    Answered,
    Asked
}

/// <summary>
/// Represents a relation (edge) in the software knowledge graph.
/// </summary>
public class Relation
{
    public int Id { get; set; }
    public RelationType Type { get; set; }
    public string Name { get; set; } = string.Empty;
    public float[] Embedding { get; set; } = Array.Empty<float>();
    
    /// <summary>
    /// Translation vector for TransH model.
    /// </summary>
    public float[] TranslationVector { get; set; } = Array.Empty<float>();
    
    /// <summary>
    /// Hyperplane normal vector for TransH model.
    /// </summary>
    public float[] HyperplaneNormal { get; set; } = Array.Empty<float>();

    public Relation() { }

    public Relation(int id, RelationType type, string? name = null)
    {
        Id = id;
        Type = type;
        Name = name ?? type.ToString();
    }

    public Relation(int id, string name, RelationType type)
    {
        Id = id;
        Name = name;
        Type = type;
    }

    public override string ToString() => $"{Name} ({Type})";
    
    public override int GetHashCode() => Id.GetHashCode();
    
    public override bool Equals(object? obj)
    {
        if (obj is Relation other)
            return Id == other.Id;
        return false;
    }
}

