namespace EPANSERec.Core.Models;

/// <summary>
/// Represents a question in the software knowledge community (e.g., StackOverflow).
/// </summary>
public class Question
{
    public int Id { get; set; }
    public string Title { get; set; } = string.Empty;
    public string Body { get; set; } = string.Empty;
    
    /// <summary>
    /// Tags/labels associated with the question.
    /// </summary>
    public List<string> Tags { get; set; } = new();
    
    /// <summary>
    /// Software knowledge entity IDs contained in this question.
    /// </summary>
    public HashSet<int> EntityIds { get; set; } = new();
    
    /// <summary>
    /// Question embedding representation e(q').
    /// </summary>
    public float[] Embedding { get; set; } = Array.Empty<float>();
    
    /// <summary>
    /// ID of the expert who answered this question (if any).
    /// </summary>
    public int? AnsweredByExpertId { get; set; }
    
    /// <summary>
    /// Whether this question has an accepted answer.
    /// </summary>
    public bool HasAcceptedAnswer { get; set; }
    
    /// <summary>
    /// Creation timestamp.
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Score/votes on the question.
    /// </summary>
    public int Score { get; set; }

    public Question() { }

    public Question(int id, string title, List<string>? tags = null)
    {
        Id = id;
        Title = title;
        Tags = tags ?? new List<string>();
    }

    public Question(int id, string title, string body)
    {
        Id = id;
        Title = title;
        Body = body;
    }

    /// <summary>
    /// Gets the number of words in the title.
    /// </summary>
    public int WordCount => Title.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
    
    /// <summary>
    /// Gets the number of entities in this question.
    /// </summary>
    public int EntityCount => EntityIds.Count;

    public override string ToString() => $"Q{Id}: {Title}";
}

