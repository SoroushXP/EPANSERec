using EPANSERec.Core.KnowledgeGraph;
using EPANSERec.Core.Models;
using EPANSERec.Core.Recommendation;
using EPANSERec.Core.Utils;

namespace EPANSERec.Tests;

/// <summary>
/// Integration tests for the EPAN-SERec model.
/// </summary>
public class EPANSERecModelTests
{
    private (SoftwareKnowledgeGraph kg, List<Expert> experts, List<Question> questions) CreateTestData()
    {
        var generator = new SampleDataGenerator(seed: 42);
        var kg = generator.GenerateKnowledgeGraph();
        var experts = generator.GenerateExperts(20, kg);
        var questions = generator.GenerateQuestions(50, kg);
        return (kg, experts, questions);
    }

    [Fact]
    public void EPANSERecModel_Creation_ShouldNotThrow()
    {
        // Arrange
        var (kg, _, _) = CreateTestData();
        
        // Act
        var model = new EPANSERecModel(kg, embeddingDim: 32, beta: 0.1f);
        
        // Assert
        Assert.NotNull(model);
    }

    [Fact]
    public void EPANSERecModel_Initialize_ShouldSetupComponents()
    {
        // Arrange
        var (kg, _, _) = CreateTestData();
        var model = new EPANSERecModel(kg, embeddingDim: 32, beta: 0.1f);
        
        // Act & Assert (should not throw)
        model.Initialize(learningRate: 0.001f);
    }

    [Fact]
    public void EPANSERecModel_PretrainKnowledgeEmbeddings_ShouldComplete()
    {
        // Arrange
        var (kg, _, _) = CreateTestData();
        var model = new EPANSERecModel(kg, embeddingDim: 32, beta: 0.1f);
        model.Initialize(learningRate: 0.001f);
        
        // Act & Assert (should not throw)
        model.PretrainKnowledgeEmbeddings(epochs: 5, batchSize: 16);
    }

    [Fact]
    public void EPANSERecModel_GenerateExpertPreferenceGraphs_ShouldComplete()
    {
        // Arrange
        var (kg, experts, _) = CreateTestData();
        var model = new EPANSERecModel(kg, embeddingDim: 32, beta: 0.1f);
        model.Initialize(learningRate: 0.001f);
        model.PretrainKnowledgeEmbeddings(epochs: 5, batchSize: 16);
        
        // Act & Assert (should not throw)
        model.GenerateExpertPreferenceGraphs(experts.Take(5), episodes: 5);
    }

    [Fact]
    public void EPANSERecModel_RecommendExperts_ShouldReturnTopK()
    {
        // Arrange
        var (kg, experts, questions) = CreateTestData();
        var model = new EPANSERecModel(kg, embeddingDim: 32, beta: 0.1f);
        model.Initialize(learningRate: 0.001f);
        model.PretrainKnowledgeEmbeddings(epochs: 5, batchSize: 16);
        model.GenerateExpertPreferenceGraphs(experts.Take(10), episodes: 3);
        model.OptimizeExpertEmbeddings(epochs: 3);
        
        // Act
        var recommendations = model.RecommendExperts(questions[0], experts, topK: 5);
        
        // Assert
        Assert.Equal(5, recommendations.Count);
        Assert.All(recommendations, r => Assert.InRange(r.score, 0, 1));
    }

    [Fact]
    public void EPANSERecModel_RecommendExperts_ShouldBeSortedByScore()
    {
        // Arrange
        var (kg, experts, questions) = CreateTestData();
        var model = new EPANSERecModel(kg, embeddingDim: 32, beta: 0.1f);
        model.Initialize(learningRate: 0.001f);
        model.PretrainKnowledgeEmbeddings(epochs: 5, batchSize: 16);
        model.GenerateExpertPreferenceGraphs(experts.Take(10), episodes: 3);
        model.OptimizeExpertEmbeddings(epochs: 3);
        
        // Act
        var recommendations = model.RecommendExperts(questions[0], experts, topK: 5);
        
        // Assert - Should be sorted descending by score
        for (int i = 0; i < recommendations.Count - 1; i++)
        {
            Assert.True(recommendations[i].score >= recommendations[i + 1].score);
        }
    }

    [Fact]
    public void EPANSERecModel_Evaluate_ShouldReturnMetrics()
    {
        // Arrange
        var generator = new SampleDataGenerator(seed: 42);
        var (kg, experts, questions, samples) = generator.GenerateDataset(20, 50, 200);
        var model = new EPANSERecModel(kg, embeddingDim: 32, beta: 0.1f);
        model.Initialize(learningRate: 0.001f);
        model.PretrainKnowledgeEmbeddings(epochs: 5, batchSize: 16);
        model.GenerateExpertPreferenceGraphs(experts, episodes: 3);
        model.OptimizeExpertEmbeddings(epochs: 3);
        
        // Prepare evaluation data
        var expertDict = experts.ToDictionary(e => e.Id);
        var questionDict = questions.ToDictionary(q => q.Id);
        var evalData = samples.Take(50)
            .Where(s => questionDict.ContainsKey(s.questionId) && expertDict.ContainsKey(s.expertId))
            .Select(s => (questionDict[s.questionId], expertDict[s.expertId], s.answered))
            .ToList();
        
        // Act
        var (auc, acc, f1) = model.Evaluate(evalData);
        
        // Assert
        Assert.InRange(auc, 0, 1);
        Assert.InRange(acc, 0, 1);
        Assert.InRange(f1, 0, 1);
    }
}

