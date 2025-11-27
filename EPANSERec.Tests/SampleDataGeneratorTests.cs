using EPANSERec.Core.Utils;

namespace EPANSERec.Tests;

/// <summary>
/// Unit tests for Sample Data Generator.
/// </summary>
public class SampleDataGeneratorTests
{
    [Fact]
    public void GenerateKnowledgeGraph_ShouldCreateEntitiesAndRelations()
    {
        // Arrange
        var generator = new SampleDataGenerator(seed: 42);
        
        // Act
        var kg = generator.GenerateKnowledgeGraph();
        
        // Assert
        Assert.True(kg.Entities.Count > 0);
        Assert.True(kg.Relations.Count > 0);
        Assert.True(kg.Triples.Count > 0);
    }

    [Fact]
    public void GenerateExperts_ShouldCreateRequestedNumber()
    {
        // Arrange
        var generator = new SampleDataGenerator(seed: 42);
        var kg = generator.GenerateKnowledgeGraph();
        
        // Act
        var experts = generator.GenerateExperts(count: 50, kg);
        
        // Assert
        Assert.Equal(50, experts.Count);
    }

    [Fact]
    public void GenerateExperts_ShouldHaveExpertiseTags()
    {
        // Arrange
        var generator = new SampleDataGenerator(seed: 42);
        var kg = generator.GenerateKnowledgeGraph();
        
        // Act
        var experts = generator.GenerateExperts(count: 10, kg);
        
        // Assert
        Assert.All(experts, e => Assert.True(e.ExpertiseTags.Count > 0));
    }

    [Fact]
    public void GenerateQuestions_ShouldCreateRequestedNumber()
    {
        // Arrange
        var generator = new SampleDataGenerator(seed: 42);
        var kg = generator.GenerateKnowledgeGraph();
        
        // Act
        var questions = generator.GenerateQuestions(count: 100, kg);
        
        // Assert
        Assert.Equal(100, questions.Count);
    }

    [Fact]
    public void GenerateQuestions_ShouldHaveTags()
    {
        // Arrange
        var generator = new SampleDataGenerator(seed: 42);
        var kg = generator.GenerateKnowledgeGraph();
        
        // Act
        var questions = generator.GenerateQuestions(count: 20, kg);
        
        // Assert
        Assert.All(questions, q => Assert.True(q.Tags.Count > 0));
    }

    [Fact]
    public void GenerateTrainingSamples_ShouldCreateRequestedNumber()
    {
        // Arrange
        var generator = new SampleDataGenerator(seed: 42);
        var kg = generator.GenerateKnowledgeGraph();
        var experts = generator.GenerateExperts(50, kg);
        var questions = generator.GenerateQuestions(100, kg);
        
        // Act
        var samples = generator.GenerateTrainingSamples(questions, experts, count: 500);
        
        // Assert
        Assert.Equal(500, samples.Count);
    }

    [Fact]
    public void GenerateTrainingSamples_ShouldHaveBalancedLabels()
    {
        // Arrange
        var generator = new SampleDataGenerator(seed: 42);
        var kg = generator.GenerateKnowledgeGraph();
        var experts = generator.GenerateExperts(100, kg);
        var questions = generator.GenerateQuestions(200, kg);
        
        // Act
        var samples = generator.GenerateTrainingSamples(questions, experts, count: 1000);
        
        // Assert
        int positives = samples.Count(s => s.answered);
        int negatives = samples.Count - positives;
        
        // Should be roughly balanced (40-60% range)
        double positiveRatio = (double)positives / samples.Count;
        Assert.True(positiveRatio >= 0.35 && positiveRatio <= 0.65, 
            $"Positive ratio {positiveRatio:P} is not balanced");
    }

    [Fact]
    public void GenerateDataset_ShouldReturnCompleteDataset()
    {
        // Arrange
        var generator = new SampleDataGenerator(seed: 42);
        
        // Act
        var (kg, experts, questions, samples) = generator.GenerateDataset(
            numExperts: 25, numQuestions: 50, numSamples: 200);
        
        // Assert
        Assert.NotNull(kg);
        Assert.Equal(25, experts.Count);
        Assert.Equal(50, questions.Count);
        Assert.Equal(200, samples.Count);
    }

    [Fact]
    public void GenerateDataset_WithSameSeed_ShouldBeReproducible()
    {
        // Arrange
        var generator1 = new SampleDataGenerator(seed: 123);
        var generator2 = new SampleDataGenerator(seed: 123);
        
        // Act
        var (kg1, experts1, _, _) = generator1.GenerateDataset(10, 20, 50);
        var (kg2, experts2, _, _) = generator2.GenerateDataset(10, 20, 50);
        
        // Assert
        Assert.Equal(kg1.Entities.Count, kg2.Entities.Count);
        Assert.Equal(experts1.Count, experts2.Count);
        Assert.Equal(experts1[0].Name, experts2[0].Name);
    }
}

