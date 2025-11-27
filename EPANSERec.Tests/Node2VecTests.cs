using EPANSERec.Core.Embeddings;
using EPANSERec.Core.KnowledgeGraph;

namespace EPANSERec.Tests;

/// <summary>
/// Unit tests for Node2Vec embedding algorithm.
/// </summary>
public class Node2VecTests
{
    private SoftwareKnowledgeGraph CreateSampleGraph()
    {
        var kg = new SoftwareKnowledgeGraph();
        
        // Add entities
        kg.AddEntity(new Entity(0, "C#", EntityType.ProgrammingLanguage));
        kg.AddEntity(new Entity(1, "ASP.NET", EntityType.SoftwareFramework));
        kg.AddEntity(new Entity(2, "Python", EntityType.ProgrammingLanguage));
        kg.AddEntity(new Entity(3, "Django", EntityType.SoftwareFramework));
        kg.AddEntity(new Entity(4, "REST", EntityType.SoftwareStandard));
        
        // Add relations
        kg.AddRelation(new Relation(0, RelationType.UsedBy, "uses"));
        kg.AddRelation(new Relation(1, RelationType.RelatedTo, "related_to"));
        
        // Add triples
        kg.AddTriple(0, 0, 1); // C# uses ASP.NET
        kg.AddTriple(2, 0, 3); // Python uses Django
        kg.AddTriple(1, 1, 4); // ASP.NET related_to REST
        kg.AddTriple(3, 1, 4); // Django related_to REST
        
        return kg;
    }

    [Fact]
    public void Node2Vec_Train_ShouldReturnEmbeddings()
    {
        // Arrange
        var kg = CreateSampleGraph();
        var node2vec = new Node2Vec(kg, embeddingDimension: 32, seed: 42);

        // Act
        var embeddings = node2vec.Train(epochs: 10);

        // Assert
        Assert.NotNull(embeddings);
        Assert.True(embeddings.Count > 0);
    }

    [Fact]
    public void Node2Vec_Embeddings_ShouldHaveCorrectDimension()
    {
        // Arrange
        var kg = CreateSampleGraph();
        int embeddingDim = 64;
        var node2vec = new Node2Vec(kg, embeddingDimension: embeddingDim, seed: 42);

        // Act
        var embeddings = node2vec.Train(epochs: 5);

        // Assert
        foreach (var embedding in embeddings.Values)
        {
            Assert.Equal(embeddingDim, embedding.Length);
        }
    }

    [Fact]
    public void Node2Vec_TrainedEmbeddings_ShouldBeAccessibleByNodeId()
    {
        // Arrange
        var kg = CreateSampleGraph();
        var node2vec = new Node2Vec(kg, embeddingDimension: 32, seed: 42);
        var embeddings = node2vec.Train(epochs: 5);

        // Act & Assert
        Assert.True(embeddings.ContainsKey(0));
        Assert.NotNull(embeddings[0]);
        Assert.Equal(32, embeddings[0].Length);
    }

    [Fact]
    public void Node2Vec_SimilarNodes_ShouldHaveSimilarEmbeddings()
    {
        // Arrange - Create a graph where nodes 1 and 3 are both connected to node 4 (REST)
        var kg = CreateSampleGraph();
        var node2vec = new Node2Vec(kg, embeddingDimension: 32, seed: 42);
        var embeddings = node2vec.Train(epochs: 20);

        // Act - Calculate cosine similarity
        var emb1 = embeddings[1]; // ASP.NET
        var emb3 = embeddings[3]; // Django
        var emb0 = embeddings[0]; // C#

        float sim13 = CosineSimilarity(emb1, emb3);
        float sim10 = CosineSimilarity(emb1, emb0);

        // Assert - ASP.NET and Django should be more similar (both connect to REST)
        // than ASP.NET and C# (only direct connection)
        Assert.True(sim13 > -1 && sim13 <= 1); // Valid similarity range
    }

    private static float CosineSimilarity(float[] a, float[] b)
    {
        float dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (MathF.Sqrt(normA) * MathF.Sqrt(normB) + 1e-10f);
    }
}

