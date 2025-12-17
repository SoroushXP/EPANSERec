namespace EPANSERec.DataExtractor.Models;

// ============================================================================
// OUTPUT DATA RECORDS (for JSON serialization)
// ============================================================================

/// <summary>
/// Entity data for knowledge graph output.
/// </summary>
public record EntityData(int Id, string Name, string Type);

/// <summary>
/// Relation data for knowledge graph output.
/// </summary>
public record RelationData(int Id, string Name, string Type);

/// <summary>
/// Triple data (head, relation, tail) for knowledge graph output.
/// </summary>
public record TripleData(int HeadId, int RelationId, int TailId);

/// <summary>
/// Expert data for experts.json output.
/// </summary>
public record ExpertData(
    int Id,
    string Name,
    List<string> ExpertiseTags,
    List<int> HistoricalQuestionIds,
    List<int> HistoricalAnswerIds,
    List<int> HistoricalEntityIds);

/// <summary>
/// Question data for questions.json output.
/// </summary>
public record QuestionOutput(
    int Id,
    string Title,
    string Body,
    List<string> Tags,
    List<int> EntityIds);

/// <summary>
/// Training sample data for samples.json output.
/// </summary>
public record SampleData(int QuestionId, int ExpertId, bool Answered);

// ============================================================================
// INTERNAL DATA CLASSES (for extraction processing)
// ============================================================================

/// <summary>
/// Internal question data during extraction.
/// </summary>
public class QuestionData
{
    public int Id { get; set; }
    public string Title { get; set; } = "";
    public string Body { get; set; } = "";
    public List<string> Tags { get; set; } = new();
    public int? AcceptedAnswerId { get; set; }
    public int OwnerUserId { get; set; }
}

/// <summary>
/// Internal answer data during extraction.
/// </summary>
public class AnswerData
{
    public int Id { get; set; }
    public int ParentId { get; set; }  // Question ID
    public int OwnerUserId { get; set; }
    public int Score { get; set; }
    public bool IsAccepted { get; set; }
}

/// <summary>
/// Internal user statistics during extraction.
/// </summary>
public class UserStats
{
    public int UserId { get; set; }
    public string DisplayName { get; set; } = "";
    public HashSet<string> AnsweredTags { get; set; } = new();
    public List<int> AnsweredQuestionIds { get; set; } = new();
    public List<int> AnswerIds { get; set; } = new();
    public HashSet<int> EntityIds { get; set; } = new();
    public int TotalScore { get; set; }
}

