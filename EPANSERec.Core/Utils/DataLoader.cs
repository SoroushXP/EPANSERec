using EPANSERec.Core.KnowledgeGraph;
using EPANSERec.Core.Models;
using System.Text.Json;

namespace EPANSERec.Core.Utils;

/// <summary>
/// Data loader for loading and preprocessing StackOverflow-like Q&A data.
/// </summary>
public class DataLoader
{
    private readonly Random _random;

    public DataLoader(int? seed = null)
    {
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Loads knowledge graph from JSON file.
    /// </summary>
    public SoftwareKnowledgeGraph LoadKnowledgeGraph(string entitiesPath, string relationsPath, string triplesPath)
    {
        var kg = new SoftwareKnowledgeGraph();
        
        // Load entities
        if (File.Exists(entitiesPath))
        {
            var entitiesJson = File.ReadAllText(entitiesPath);
            var entities = JsonSerializer.Deserialize<List<EntityData>>(entitiesJson);
            foreach (var e in entities ?? new List<EntityData>())
            {
                var entity = new Entity(e.Id, e.Name, Enum.Parse<EntityType>(e.Type));
                kg.AddEntity(entity);
            }
        }
        
        // Load relations
        if (File.Exists(relationsPath))
        {
            var relationsJson = File.ReadAllText(relationsPath);
            var relations = JsonSerializer.Deserialize<List<RelationData>>(relationsJson);
            foreach (var r in relations ?? new List<RelationData>())
            {
                var relation = new Relation(r.Id, Enum.Parse<RelationType>(r.Type), r.Name);
                kg.AddRelation(relation);
            }
        }
        
        // Load triples
        if (File.Exists(triplesPath))
        {
            var triplesJson = File.ReadAllText(triplesPath);
            var triples = JsonSerializer.Deserialize<List<TripleData>>(triplesJson);
            foreach (var t in triples ?? new List<TripleData>())
            {
                kg.AddTriple(t.HeadId, t.RelationId, t.TailId);
            }
        }
        
        return kg;
    }

    /// <summary>
    /// Loads experts from JSON file.
    /// </summary>
    public List<Expert> LoadExperts(string path)
    {
        if (!File.Exists(path)) return new List<Expert>();
        
        var json = File.ReadAllText(path);
        var expertsData = JsonSerializer.Deserialize<List<ExpertData>>(json);
        
        return expertsData?.Select(e => {
            var expert = new Expert(e.Id, e.Name);
            if (e.ExpertiseTags != null)
                foreach (var tag in e.ExpertiseTags) expert.ExpertiseTags.Add(tag);
            if (e.HistoricalQuestionIds != null)
                foreach (var id in e.HistoricalQuestionIds) expert.HistoricalQuestionIds.Add(id);
            if (e.HistoricalAnswerIds != null)
                foreach (var id in e.HistoricalAnswerIds) expert.HistoricalAnswerIds.Add(id);
            if (e.HistoricalEntityIds != null)
                foreach (var id in e.HistoricalEntityIds) expert.HistoricalEntityIds.Add(id);
            return expert;
        }).ToList() ?? new List<Expert>();
    }

    /// <summary>
    /// Loads questions from JSON file.
    /// </summary>
    public List<Question> LoadQuestions(string path)
    {
        if (!File.Exists(path)) return new List<Question>();
        
        var json = File.ReadAllText(path);
        var questionsData = JsonSerializer.Deserialize<List<QuestionData>>(json);
        
        return questionsData?.Select(q => {
            var question = new Question(q.Id, q.Title, q.Body);
            if (q.Tags != null)
                question.Tags = q.Tags.ToList();
            if (q.EntityIds != null)
                foreach (var id in q.EntityIds) question.EntityIds.Add(id);
            return question;
        }).ToList() ?? new List<Question>();
    }

    /// <summary>
    /// Loads training samples (question-expert-label triplets).
    /// </summary>
    public List<(int questionId, int expertId, bool answered)> LoadTrainingSamples(string path)
    {
        if (!File.Exists(path)) return new List<(int, int, bool)>();
        
        var json = File.ReadAllText(path);
        var samples = JsonSerializer.Deserialize<List<SampleData>>(json);
        
        return samples?.Select(s => (s.QuestionId, s.ExpertId, s.Answered)).ToList() 
            ?? new List<(int, int, bool)>();
    }

    /// <summary>
    /// Splits data into train/validation/test sets.
    /// </summary>
    public (List<T> train, List<T> val, List<T> test) SplitData<T>(
        List<T> data, float trainRatio = 0.8f, float valRatio = 0.1f)
    {
        var shuffled = data.OrderBy(_ => _random.Next()).ToList();
        int trainSize = (int)(data.Count * trainRatio);
        int valSize = (int)(data.Count * valRatio);
        
        var train = shuffled.Take(trainSize).ToList();
        var val = shuffled.Skip(trainSize).Take(valSize).ToList();
        var test = shuffled.Skip(trainSize + valSize).ToList();
        
        return (train, val, test);
    }

    // Data transfer objects for JSON deserialization
    private record EntityData(int Id, string Name, string Type);
    private record RelationData(int Id, string Name, string Type);
    private record TripleData(int HeadId, int RelationId, int TailId);
    private record ExpertData(int Id, string Name, List<string>? ExpertiseTags, 
        List<int>? HistoricalQuestionIds, List<int>? HistoricalAnswerIds, List<int>? HistoricalEntityIds);
    private record QuestionData(int Id, string Title, string Body, List<string>? Tags, List<int>? EntityIds);
    private record SampleData(int QuestionId, int ExpertId, bool Answered);
}

