using EPANSERec.Core.KnowledgeGraph;

namespace EPANSERec.Tests;

/// <summary>
/// Unit tests for Knowledge Graph components.
/// </summary>
public class KnowledgeGraphTests
{
    [Fact]
    public void Entity_Creation_ShouldSetProperties()
    {
        // Arrange & Act
        var entity = new Entity(1, "C#", EntityType.ProgrammingLanguage);
        
        // Assert
        Assert.Equal(1, entity.Id);
        Assert.Equal("C#", entity.Name);
        Assert.Equal(EntityType.ProgrammingLanguage, entity.Type);
    }

    [Fact]
    public void Relation_Creation_ShouldSetProperties()
    {
        // Arrange & Act
        var relation = new Relation(1, RelationType.UsedBy, "uses");
        
        // Assert
        Assert.Equal(1, relation.Id);
        Assert.Equal(RelationType.UsedBy, relation.Type);
        Assert.Equal("uses", relation.Name);
    }

    [Fact]
    public void Triple_Creation_ShouldSetEntitiesAndRelation()
    {
        // Arrange
        var head = new Entity(1, "C#", EntityType.ProgrammingLanguage);
        var tail = new Entity(2, "ASP.NET", EntityType.SoftwareFramework);
        var relation = new Relation(1, RelationType.UsedBy, "uses");
        
        // Act
        var triple = new Triple(head, relation, tail);
        
        // Assert
        Assert.Equal(head, triple.Head);
        Assert.Equal(tail, triple.Tail);
        Assert.Equal(relation, triple.Relation);
    }

    [Fact]
    public void SoftwareKnowledgeGraph_AddEntity_ShouldAddToEntities()
    {
        // Arrange
        var kg = new SoftwareKnowledgeGraph();
        var entity = new Entity(1, "Python", EntityType.ProgrammingLanguage);
        
        // Act
        kg.AddEntity(entity);
        
        // Assert
        Assert.Single(kg.Entities);
        Assert.True(kg.Entities.ContainsKey(1));
        Assert.Equal("Python", kg.Entities[1].Name);
    }

    [Fact]
    public void SoftwareKnowledgeGraph_AddRelation_ShouldAddToRelations()
    {
        // Arrange
        var kg = new SoftwareKnowledgeGraph();
        var relation = new Relation(1, RelationType.DependsOn, "depends_on");
        
        // Act
        kg.AddRelation(relation);
        
        // Assert
        Assert.Single(kg.Relations);
        Assert.True(kg.Relations.ContainsKey(1));
    }

    [Fact]
    public void SoftwareKnowledgeGraph_AddTriple_ShouldAddToTriples()
    {
        // Arrange
        var kg = new SoftwareKnowledgeGraph();
        var head = new Entity(1, "C#", EntityType.ProgrammingLanguage);
        var tail = new Entity(2, "ASP.NET", EntityType.SoftwareFramework);
        var relation = new Relation(1, RelationType.UsedBy, "uses");
        kg.AddEntity(head);
        kg.AddEntity(tail);
        kg.AddRelation(relation);
        
        // Act
        var triple = new Triple(head, relation, tail);
        kg.AddTriple(triple);
        
        // Assert
        Assert.Single(kg.Triples);
    }

    [Fact]
    public void SoftwareKnowledgeGraph_GetNeighbors_ShouldReturnConnectedNodes()
    {
        // Arrange
        var kg = new SoftwareKnowledgeGraph();
        var entity1 = new Entity(1, "C#", EntityType.ProgrammingLanguage);
        var entity2 = new Entity(2, "ASP.NET", EntityType.SoftwareFramework);
        var relation = new Relation(1, RelationType.UsedBy, "uses");
        kg.AddEntity(entity1);
        kg.AddEntity(entity2);
        kg.AddRelation(relation);
        kg.AddTriple(new Triple(entity1, relation, entity2));

        // Act
        var neighbors = kg.GetNeighbors(1);

        // Assert
        Assert.Single(neighbors);
        Assert.Equal(2, neighbors[0].neighbor.Id);
    }

    [Fact]
    public void SoftwareKnowledgeGraph_AddTripleByIds_ShouldCreateConnection()
    {
        // Arrange
        var kg = new SoftwareKnowledgeGraph();
        kg.AddEntity(new Entity(1, "Java", EntityType.ProgrammingLanguage));
        kg.AddEntity(new Entity(2, "Spring", EntityType.SoftwareFramework));
        kg.AddRelation(new Relation(1, RelationType.UsedBy, "uses"));

        // Act
        kg.AddTriple(1, 1, 2);

        // Assert
        var neighbors = kg.GetNeighbors(1);
        Assert.Contains(neighbors, n => n.neighbor.Id == 2);
    }
}

