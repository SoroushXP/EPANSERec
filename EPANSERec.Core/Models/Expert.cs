namespace EPANSERec.Core.Models;

/// <summary>
/// Represents an expert user in the software knowledge community.
/// </summary>
public class Expert
{
    public int Id { get; set; }
    public string Username { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public int ReputationScore { get; set; }

    /// <summary>
    /// Historical Q&A set of the expert (qui in the paper).
    /// </summary>
    public List<Question> HistoricalQA { get; set; } = new();

    /// <summary>
    /// Entity set from expert's historical Q&A (Eu in the paper).
    /// </summary>
    public HashSet<int> HistoricalEntityIds { get; set; } = new();

    /// <summary>
    /// Historical question IDs the expert has answered.
    /// </summary>
    public HashSet<int> HistoricalQuestionIds { get; set; } = new();

    /// <summary>
    /// Historical answer IDs from the expert.
    /// </summary>
    public HashSet<int> HistoricalAnswerIds { get; set; } = new();

    /// <summary>
    /// Expert's embedding representation e(ui).
    /// </summary>
    public float[] Embedding { get; set; } = Array.Empty<float>();

    /// <summary>
    /// Expertise preference embedding derived from DRL.
    /// </summary>
    public float[] ExpertisePreferenceEmbedding { get; set; } = Array.Empty<float>();

    /// <summary>
    /// Labels/tags that represent the expert's areas of expertise.
    /// </summary>
    public HashSet<string> ExpertiseTags { get; set; } = new();

    public Expert() { }

    public Expert(int id, string username, int reputationScore = 0)
    {
        Id = id;
        Username = username;
        Name = username;
        ReputationScore = reputationScore;
    }

    public Expert(int id, string name)
    {
        Id = id;
        Name = name;
        Username = name;
    }

    /// <summary>
    /// Adds a question to the expert's historical Q&A and updates entity set.
    /// </summary>
    public void AddHistoricalQuestion(Question question)
    {
        HistoricalQA.Add(question);
        foreach (var entityId in question.EntityIds)
        {
            HistoricalEntityIds.Add(entityId);
        }
    }

    /// <summary>
    /// Gets the number of historical Q&A items.
    /// </summary>
    public int HistoricalQACount => HistoricalQA.Count;

    public override string ToString() => $"{Username} (Rep: {ReputationScore})";
}

