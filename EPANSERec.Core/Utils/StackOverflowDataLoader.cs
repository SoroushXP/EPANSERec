using EPANSERec.Core.KnowledgeGraph;
using EPANSERec.Core.Models;
using System.Text.Json;

namespace EPANSERec.Core.Utils;

/// <summary>
/// Data loader for real StackOverflow dataset.
/// 
/// Expected directory structure:
/// {DataPath}/
///   ├── knowledge_graph/
///   │   ├── entities.json       - List of software entities (tags, technologies)
///   │   ├── relations.json      - Relation type definitions
///   │   └── triples.json        - Entity-relation-entity triples
///   ├── experts.json            - User data with expertise info
///   ├── questions.json          - Question data with tags
///   └── samples.json            - Training samples (question, expert, answered)
/// 
/// File Formats:
/// 
/// entities.json:
///   [{"id": 1, "name": "python", "type": "ProgrammingLanguage"}, ...]
/// 
/// relations.json:
///   [{"id": 1, "name": "related_to", "type": "RelatedTo"}, ...]
/// 
/// triples.json:
///   [{"headId": 1, "relationId": 1, "tailId": 2}, ...]
/// 
/// experts.json:
///   [{"id": 1, "name": "user123", "expertiseTags": ["python", "django"], 
///     "historicalQuestionIds": [1, 2], "historicalAnswerIds": [10, 20],
///     "historicalEntityIds": [1, 5]}, ...]
/// 
/// questions.json:
///   [{"id": 1, "title": "How to...", "body": "...", "tags": ["python"], "entityIds": [1]}, ...]
/// 
/// samples.json:
///   [{"questionId": 1, "expertId": 1, "answered": true}, ...]
/// </summary>
public class StackOverflowDataLoader
{
    private readonly string _basePath;
    private readonly Random _random;

    public StackOverflowDataLoader(string basePath, int? seed = null)
    {
        _basePath = basePath;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Loads the complete StackOverflow dataset.
    /// </summary>
    public (SoftwareKnowledgeGraph kg, List<Expert> experts, List<Question> questions,
        List<(int questionId, int expertId, bool answered)> samples) LoadDataset()
    {
        Console.WriteLine($"Loading StackOverflow dataset from: {_basePath}");
        
        var kg = LoadKnowledgeGraph();
        var experts = LoadExperts();
        var questions = LoadQuestions();
        var samples = LoadSamples();
        
        return (kg, experts, questions, samples);
    }

    /// <summary>
    /// Loads the software knowledge graph from JSON files.
    /// </summary>
    public SoftwareKnowledgeGraph LoadKnowledgeGraph()
    {
        var kg = new SoftwareKnowledgeGraph();
        var kgPath = Path.Combine(_basePath, "knowledge_graph");
        
        // Load entities
        var entitiesPath = Path.Combine(kgPath, "entities.json");
        if (File.Exists(entitiesPath))
        {
            var json = File.ReadAllText(entitiesPath);
            var entities = JsonSerializer.Deserialize<List<EntityData>>(json, JsonOptions);
            foreach (var e in entities ?? [])
            {
                if (Enum.TryParse<EntityType>(e.Type, true, out var entityType))
                {
                    kg.AddEntity(new Entity(e.Id, e.Name, entityType));
                }
            }
            Console.WriteLine($"  Loaded {kg.Entities.Count} entities");
        }
        else
        {
            Console.WriteLine($"  Warning: entities.json not found at {entitiesPath}");
        }
        
        // Load relations
        var relationsPath = Path.Combine(kgPath, "relations.json");
        if (File.Exists(relationsPath))
        {
            var json = File.ReadAllText(relationsPath);
            var relations = JsonSerializer.Deserialize<List<RelationData>>(json, JsonOptions);
            foreach (var r in relations ?? [])
            {
                if (Enum.TryParse<RelationType>(r.Type, true, out var relType))
                {
                    kg.AddRelation(new Relation(r.Id, relType, r.Name));
                }
            }
            Console.WriteLine($"  Loaded {kg.Relations.Count} relations");
        }
        
        // Load triples
        var triplesPath = Path.Combine(kgPath, "triples.json");
        if (File.Exists(triplesPath))
        {
            var json = File.ReadAllText(triplesPath);
            var triples = JsonSerializer.Deserialize<List<TripleData>>(json, JsonOptions);
            foreach (var t in triples ?? [])
            {
                kg.AddTriple(t.HeadId, t.RelationId, t.TailId);
            }
            Console.WriteLine($"  Loaded {kg.Triples.Count} triples");
        }
        
        return kg;
    }

    /// <summary>
    /// Loads expert/user data from JSON file.
    /// </summary>
    public List<Expert> LoadExperts()
    {
        var path = Path.Combine(_basePath, "experts.json");
        if (!File.Exists(path))
        {
            Console.WriteLine($"  Warning: experts.json not found at {path}");
            return [];
        }
        
        var json = File.ReadAllText(path);
        var expertsData = JsonSerializer.Deserialize<List<ExpertData>>(json, JsonOptions);
        
        var experts = expertsData?.Select(e =>
        {
            var expert = new Expert(e.Id, e.Name);
            foreach (var tag in e.ExpertiseTags ?? []) expert.ExpertiseTags.Add(tag);
            foreach (var id in e.HistoricalQuestionIds ?? []) expert.HistoricalQuestionIds.Add(id);
            foreach (var id in e.HistoricalAnswerIds ?? []) expert.HistoricalAnswerIds.Add(id);
            foreach (var id in e.HistoricalEntityIds ?? []) expert.HistoricalEntityIds.Add(id);
            return expert;
        }).ToList() ?? [];
        
        Console.WriteLine($"  Loaded {experts.Count} experts");
        return experts;
    }

    /// <summary>
    /// Loads question data from JSON file.
    /// </summary>
    public List<Question> LoadQuestions()
    {
        var path = Path.Combine(_basePath, "questions.json");
        if (!File.Exists(path))
        {
            Console.WriteLine($"  Warning: questions.json not found at {path}");
            return [];
        }

        var json = File.ReadAllText(path);
        var questionsData = JsonSerializer.Deserialize<List<QuestionData>>(json, JsonOptions);

        var questions = questionsData?.Select(q =>
        {
            var question = new Question(q.Id, q.Title, q.Body);
            question.Tags = q.Tags?.ToList() ?? [];
            foreach (var id in q.EntityIds ?? []) question.EntityIds.Add(id);
            return question;
        }).ToList() ?? [];

        Console.WriteLine($"  Loaded {questions.Count} questions");
        return questions;
    }

    /// <summary>
    /// Loads training samples from JSON file.
    /// </summary>
    public List<(int questionId, int expertId, bool answered)> LoadSamples()
    {
        var path = Path.Combine(_basePath, "samples.json");
        if (!File.Exists(path))
        {
            Console.WriteLine($"  Warning: samples.json not found at {path}");
            return [];
        }

        var json = File.ReadAllText(path);
        var samplesData = JsonSerializer.Deserialize<List<SampleData>>(json, JsonOptions);

        var samples = samplesData?.Select(s => (s.QuestionId, s.ExpertId, s.Answered)).ToList() ?? [];

        Console.WriteLine($"  Loaded {samples.Count} training samples");
        return samples;
    }

    /// <summary>
    /// Checks if the dataset exists at the specified path.
    /// </summary>
    public bool DatasetExists()
    {
        var kgPath = Path.Combine(_basePath, "knowledge_graph");
        return Directory.Exists(kgPath) &&
               File.Exists(Path.Combine(kgPath, "entities.json")) &&
               File.Exists(Path.Combine(_basePath, "experts.json")) &&
               File.Exists(Path.Combine(_basePath, "questions.json"));
    }

    /// <summary>
    /// Gets dataset statistics without fully loading the data.
    /// </summary>
    public DatasetStats GetStats()
    {
        var stats = new DatasetStats();

        var entitiesPath = Path.Combine(_basePath, "knowledge_graph", "entities.json");
        if (File.Exists(entitiesPath))
        {
            var json = File.ReadAllText(entitiesPath);
            var entities = JsonSerializer.Deserialize<List<EntityData>>(json, JsonOptions);
            stats.EntityCount = entities?.Count ?? 0;
        }

        var triplesPath = Path.Combine(_basePath, "knowledge_graph", "triples.json");
        if (File.Exists(triplesPath))
        {
            var json = File.ReadAllText(triplesPath);
            var triples = JsonSerializer.Deserialize<List<TripleData>>(json, JsonOptions);
            stats.TripleCount = triples?.Count ?? 0;
        }

        var expertsPath = Path.Combine(_basePath, "experts.json");
        if (File.Exists(expertsPath))
        {
            var json = File.ReadAllText(expertsPath);
            var experts = JsonSerializer.Deserialize<List<ExpertData>>(json, JsonOptions);
            stats.ExpertCount = experts?.Count ?? 0;
        }

        var questionsPath = Path.Combine(_basePath, "questions.json");
        if (File.Exists(questionsPath))
        {
            var json = File.ReadAllText(questionsPath);
            var questions = JsonSerializer.Deserialize<List<QuestionData>>(json, JsonOptions);
            stats.QuestionCount = questions?.Count ?? 0;
        }

        var samplesPath = Path.Combine(_basePath, "samples.json");
        if (File.Exists(samplesPath))
        {
            var json = File.ReadAllText(samplesPath);
            var samples = JsonSerializer.Deserialize<List<SampleData>>(json, JsonOptions);
            stats.SampleCount = samples?.Count ?? 0;
        }

        return stats;
    }

    // JSON serialization options (case-insensitive)
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    // Data transfer objects
    private record EntityData(int Id, string Name, string Type);
    private record RelationData(int Id, string Name, string Type);
    private record TripleData(int HeadId, int RelationId, int TailId);
    private record ExpertData(int Id, string Name, List<string>? ExpertiseTags,
        List<int>? HistoricalQuestionIds, List<int>? HistoricalAnswerIds, List<int>? HistoricalEntityIds);
    private record QuestionData(int Id, string Title, string Body, List<string>? Tags, List<int>? EntityIds);
    private record SampleData(int QuestionId, int ExpertId, bool Answered);
}

/// <summary>
/// Dataset statistics for quick overview.
/// </summary>
public class DatasetStats
{
    public int EntityCount { get; set; }
    public int TripleCount { get; set; }
    public int ExpertCount { get; set; }
    public int QuestionCount { get; set; }
    public int SampleCount { get; set; }

    public override string ToString() =>
        $"Entities: {EntityCount}, Triples: {TripleCount}, Experts: {ExpertCount}, " +
        $"Questions: {QuestionCount}, Samples: {SampleCount}";
}

