using EPANSERec.Core.Embeddings;
using EPANSERec.Core.KnowledgeGraph;

namespace EPANSERec.Tests;

/// <summary>
/// Unit tests for TransH knowledge embedding model.
/// </summary>
public class TransHTests
{
    private SoftwareKnowledgeGraph CreateSampleGraph()
    {
        var kg = new SoftwareKnowledgeGraph();

        // Add entities
        for (int i = 0; i < 20; i++)
        {
            kg.AddEntity(new Entity(i, $"Entity_{i}", EntityType.ProgrammingLanguage));
        }

        // Add relations
        for (int i = 0; i < 5; i++)
        {
            kg.AddRelation(new Relation(i, RelationType.UsedBy, $"relation_{i}"));
        }

        // Add triples
        kg.AddTriple(0, 0, 1);
        kg.AddTriple(1, 1, 2);
        kg.AddTriple(2, 0, 3);
        kg.AddTriple(3, 2, 4);
        kg.AddTriple(4, 1, 5);

        return kg;
    }

    [Fact]
    public void TransH_Creation_ShouldInitializeCorrectly()
    {
        // Arrange & Act
        var transH = new TransH(numEntities: 100, numRelations: 10, embeddingDim: 64);

        // Assert
        Assert.NotNull(transH);
    }

    [Fact]
    public void TransH_GetEntityEmbedding_ShouldReturnCorrectDimension()
    {
        // Arrange
        int embeddingDim = 64;
        var transH = new TransH(numEntities: 100, numRelations: 10, embeddingDim: embeddingDim);

        // Act
        var embedding = transH.GetEntityEmbedding(0);

        // Assert
        Assert.NotNull(embedding);
        Assert.Equal(embeddingDim, embedding.Length);
    }

    [Fact]
    public void TransH_GetRelationEmbedding_ShouldReturnCorrectDimension()
    {
        // Arrange
        int embeddingDim = 64;
        var transH = new TransH(numEntities: 100, numRelations: 10, embeddingDim: embeddingDim);

        // Act
        var embedding = transH.GetRelationEmbedding(0);

        // Assert
        Assert.NotNull(embedding);
        Assert.Equal(embeddingDim, embedding.Length);
    }

    [Fact]
    public void TransH_Train_ShouldCompleteWithoutError()
    {
        // Arrange
        var kg = CreateSampleGraph();
        var transH = new TransH(numEntities: 20, numRelations: 5, embeddingDim: 32);

        // Act & Assert (should not throw)
        transH.Train(kg, epochs: 5, batchSize: 4, learningRate: 0.01f);
    }

    [Fact]
    public void TransH_AfterTraining_EmbeddingsShouldBeUpdated()
    {
        // Arrange
        var kg = CreateSampleGraph();
        var transH = new TransH(numEntities: 20, numRelations: 5, embeddingDim: 32);

        // Get initial embedding
        var initialEmb = transH.GetEntityEmbedding(0).ToArray();

        // Act
        transH.Train(kg, epochs: 10, batchSize: 4, learningRate: 0.1f);
        var trainedEmb = transH.GetEntityEmbedding(0);

        // Assert - Embeddings should have changed after training
        bool hasChanged = false;
        for (int i = 0; i < initialEmb.Length; i++)
        {
            if (Math.Abs(initialEmb[i] - trainedEmb[i]) > 1e-6)
            {
                hasChanged = true;
                break;
            }
        }
        Assert.True(hasChanged);
    }

    [Fact]
    public void TransH_DifferentEntities_ShouldHaveDifferentEmbeddings()
    {
        // Arrange
        var transH = new TransH(numEntities: 100, numRelations: 10, embeddingDim: 64);

        // Act
        var emb0 = transH.GetEntityEmbedding(0);
        var emb1 = transH.GetEntityEmbedding(1);

        // Assert - Embeddings should be different (initialized randomly)
        bool areDifferent = false;
        for (int i = 0; i < emb0.Length; i++)
        {
            if (Math.Abs(emb0[i] - emb1[i]) > 1e-6)
            {
                areDifferent = true;
                break;
            }
        }
        Assert.True(areDifferent);
    }
}

