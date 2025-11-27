using EPANSERec.Core.Models;

namespace EPANSERec.Tests;

/// <summary>
/// Unit tests for Expert and Question models.
/// </summary>
public class ModelsTests
{
    [Fact]
    public void Expert_Creation_ShouldSetProperties()
    {
        // Arrange & Act
        var expert = new Expert(1, "john_dev", 1500);
        
        // Assert
        Assert.Equal(1, expert.Id);
        Assert.Equal("john_dev", expert.Username);
        Assert.Equal(1500, expert.ReputationScore);
    }

    [Fact]
    public void Expert_ExpertiseTags_ShouldBeModifiable()
    {
        // Arrange
        var expert = new Expert(1, "jane_dev");
        
        // Act
        expert.ExpertiseTags.Add("C#");
        expert.ExpertiseTags.Add("ASP.NET");
        
        // Assert
        Assert.Equal(2, expert.ExpertiseTags.Count);
        Assert.Contains("C#", expert.ExpertiseTags);
        Assert.Contains("ASP.NET", expert.ExpertiseTags);
    }

    [Fact]
    public void Expert_HistoricalEntityIds_ShouldBeModifiable()
    {
        // Arrange
        var expert = new Expert(1, "dev_user");
        
        // Act
        expert.HistoricalEntityIds.Add(1);
        expert.HistoricalEntityIds.Add(2);
        expert.HistoricalEntityIds.Add(3);
        
        // Assert
        Assert.Equal(3, expert.HistoricalEntityIds.Count);
    }

    [Fact]
    public void Expert_HistoricalQuestionIds_ShouldBeModifiable()
    {
        // Arrange
        var expert = new Expert(1, "dev_user");
        
        // Act
        expert.HistoricalQuestionIds.Add(100);
        expert.HistoricalQuestionIds.Add(200);
        
        // Assert
        Assert.Equal(2, expert.HistoricalQuestionIds.Count);
        Assert.Contains(100, expert.HistoricalQuestionIds);
    }

    [Fact]
    public void Question_Creation_ShouldSetProperties()
    {
        // Arrange & Act
        var question = new Question(1, "How to use async/await?", "Body text here");
        
        // Assert
        Assert.Equal(1, question.Id);
        Assert.Equal("How to use async/await?", question.Title);
        Assert.Equal("Body text here", question.Body);
    }

    [Fact]
    public void Question_Tags_ShouldBeModifiable()
    {
        // Arrange
        var question = new Question(1, "Test question");
        
        // Act
        question.Tags.Add("C#");
        question.Tags.Add("async");
        question.Tags.Add("threading");
        
        // Assert
        Assert.Equal(3, question.Tags.Count);
        Assert.Contains("async", question.Tags);
    }

    [Fact]
    public void Question_EntityIds_ShouldBeModifiable()
    {
        // Arrange
        var question = new Question(1, "Test question");
        
        // Act
        question.EntityIds.Add(1);
        question.EntityIds.Add(2);
        
        // Assert
        Assert.Equal(2, question.EntityIds.Count);
    }

    [Fact]
    public void Question_ConstructorWithTags_ShouldInitializeTags()
    {
        // Arrange
        var tags = new List<string> { "python", "django" };
        
        // Act
        var question = new Question(1, "Django question", tags);
        
        // Assert
        Assert.Equal(2, question.Tags.Count);
        Assert.Contains("python", question.Tags);
        Assert.Contains("django", question.Tags);
    }
}

