namespace EPANSERec.Core.KnowledgeGraph;

/// <summary>
/// Represents an entity type in the software knowledge graph.
/// Based on SWKG: 8 entity types defined in the paper.
/// </summary>
public enum EntityType
{
    ProgrammingLanguage,
    SoftwarePlatform,
    SoftwareAPI,
    SoftwareTool,
    SoftwareLibrary,
    SoftwareFramework,
    SoftwareStandard,
    SoftwareDevelopmentProcess,
    // Additional types for flexibility
    Framework,
    Concept,
    Tool,
    Library,
    User,
    Question,
    Answer,
    Tag
}

/// <summary>
/// Represents an entity (node) in the software knowledge graph.
/// </summary>
public class Entity
{
    public int Id { get; set; }
    public string Name { get; set; } = string.Empty;
    public EntityType Type { get; set; }
    public float[] Embedding { get; set; } = Array.Empty<float>();
    
    /// <summary>
    /// Dictionary of properties associated with the entity.
    /// </summary>
    public Dictionary<string, object> Properties { get; set; } = new();

    public Entity() { }

    public Entity(int id, string name, EntityType type)
    {
        Id = id;
        Name = name;
        Type = type;
    }

    public override string ToString() => $"{Name} ({Type})";
    
    public override int GetHashCode() => Id.GetHashCode();
    
    public override bool Equals(object? obj)
    {
        if (obj is Entity other)
            return Id == other.Id;
        return false;
    }
}

