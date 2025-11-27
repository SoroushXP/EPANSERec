namespace EPANSERec.Core.KnowledgeGraph;

/// <summary>
/// Represents a knowledge graph triple (h, r, t) - head, relation, tail.
/// </summary>
public class Triple
{
    public Entity Head { get; set; }
    public Relation Relation { get; set; }
    public Entity Tail { get; set; }

    public Triple(Entity head, Relation relation, Entity tail)
    {
        Head = head;
        Relation = relation;
        Tail = tail;
    }

    /// <summary>
    /// Creates a negative sample by replacing head or tail entity.
    /// </summary>
    public Triple CreateNegativeSample(Entity replacement, bool replaceHead)
    {
        return replaceHead 
            ? new Triple(replacement, Relation, Tail) 
            : new Triple(Head, Relation, replacement);
    }

    public override string ToString() => $"({Head.Name}, {Relation.Name}, {Tail.Name})";
    
    public override int GetHashCode() => HashCode.Combine(Head.Id, Relation.Id, Tail.Id);
    
    public override bool Equals(object? obj)
    {
        if (obj is Triple other)
            return Head.Id == other.Head.Id && 
                   Relation.Id == other.Relation.Id && 
                   Tail.Id == other.Tail.Id;
        return false;
    }
}

