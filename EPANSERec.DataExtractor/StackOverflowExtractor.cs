using System.Text.Json;
using System.Xml.Linq;
using EPANSERec.DataExtractor.Models;

namespace EPANSERec.DataExtractor;

/// <summary>
/// Extracts and transforms StackOverflow data dump into EPAN-SERec format.
/// </summary>
public class StackOverflowExtractor
{
    private readonly ExtractorConfig _config;
    private readonly Dictionary<string, int> _tagToEntityId = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<int, UserStats> _userStats = new();
    private readonly Dictionary<int, QuestionData> _questions = new();
    private readonly List<AnswerData> _answers = new();
    private readonly List<EntityData> _entities = new();
    private readonly List<TripleData> _triples = new();
    private int _nextEntityId = 0;

    public StackOverflowExtractor(ExtractorConfig config)
    {
        _config = config;
    }

    /// <summary>
    /// Executes the full extraction pipeline.
    /// </summary>
    public async Task ExtractAsync()
    {
        Console.WriteLine("=== Step 1: Extracting Tags (Entities) ===");
        await ExtractTagsAsync();

        Console.WriteLine("\n=== Step 2: Extracting Posts (Questions & Answers) ===");
        await ExtractPostsAsync();

        Console.WriteLine("\n=== Step 3: Building Knowledge Graph Relations ===");
        BuildKnowledgeGraph();

        Console.WriteLine("\n=== Step 4: Saving Output Files ===");
        await SaveOutputAsync();
    }

    private async Task ExtractTagsAsync()
    {
        if (!File.Exists(_config.TagsXmlPath))
        {
            Console.WriteLine("  Tags.xml not found, using tags from Posts.xml");
            return;
        }

        Console.WriteLine($"  Reading {_config.TagsXmlPath}...");
        int count = 0;

        using var reader = new StreamReader(_config.TagsXmlPath);
        while (!reader.EndOfStream)
        {
            var line = await reader.ReadLineAsync();
            if (line == null || !line.TrimStart().StartsWith("<row")) continue;

            try
            {
                var element = XElement.Parse(line);
                var tagName = element.Attribute("TagName")?.Value?.ToLowerInvariant();
                var tagCount = int.Parse(element.Attribute("Count")?.Value ?? "0");

                if (tagName != null && tagCount >= _config.MinTagOccurrences)
                {
                    var softwareTags = _config.GetSoftwareTagsSet();
                    if (softwareTags.Count == 0 || softwareTags.Contains(tagName))
                    {
                        if (!_tagToEntityId.ContainsKey(tagName))
                        {
                            var entityType = ClassifyTag(tagName);
                            _entities.Add(new EntityData(_nextEntityId, tagName, entityType));
                            _tagToEntityId[tagName] = _nextEntityId++;
                            count++;
                        }
                    }
                }
            }
            catch { /* Skip malformed lines */ }
        }

        Console.WriteLine($"  Extracted {count} tag entities");
    }

    private static string ClassifyTag(string tag)
    {
        var languages = new[] { "python", "javascript", "java", "c#", "c++", "php", "ruby", "go",
            "rust", "swift", "kotlin", "typescript", "scala", "r", "perl", "haskell", "lua" };
        var frameworks = new[] { "django", "flask", "react", "angular", "vue", "node", "express",
            "spring", "asp.net", ".net", "rails", "laravel", "symfony" };
        var databases = new[] { "sql", "mysql", "postgresql", "mongodb", "sqlite", "redis",
            "elasticsearch", "oracle", "cassandra", "firebase" };
        var tools = new[] { "git", "docker", "kubernetes", "aws", "azure", "linux", "nginx" };

        if (languages.Any(l => tag.Contains(l))) return "ProgrammingLanguage";
        if (frameworks.Any(f => tag.Contains(f))) return "SoftwareFramework";
        if (databases.Any(d => tag.Contains(d))) return "SoftwareTool";
        if (tools.Any(t => tag.Contains(t))) return "Tool";
        return "Concept";
    }

    private async Task ExtractPostsAsync()
    {
        Console.WriteLine($"  Reading {_config.PostsXmlPath}...");
        int questionCount = 0, answerCount = 0;

        using var reader = new StreamReader(_config.PostsXmlPath);
        while (!reader.EndOfStream)
        {
            var line = await reader.ReadLineAsync();
            if (line == null || !line.TrimStart().StartsWith("<row")) continue;

            try
            {
                var element = XElement.Parse(line);
                var postType = int.Parse(element.Attribute("PostTypeId")?.Value ?? "0");

                if (postType == 1) // Question
                {
                    if (_config.MaxQuestions > 0 && questionCount >= _config.MaxQuestions) continue;
                    ProcessQuestion(element, ref questionCount);
                }
                else if (postType == 2) // Answer
                {
                    ProcessAnswer(element, ref answerCount);
                }
            }
            catch { /* Skip malformed lines */ }
        }

        Console.WriteLine($"  Extracted {questionCount} questions, {answerCount} answers");
        Console.WriteLine($"  Found {_userStats.Count} users with answers");
    }

    private void ProcessQuestion(XElement element, ref int questionCount)
    {
        var tags = ParseTags(element.Attribute("Tags")?.Value ?? "");
        var entityIds = tags.Where(t => _tagToEntityId.ContainsKey(t))
                           .Select(t => _tagToEntityId[t]).ToList();

        // Add tags as entities if not already present
        foreach (var tag in tags)
        {
            if (!_tagToEntityId.ContainsKey(tag))
            {
                var entityType = ClassifyTag(tag);
                _entities.Add(new EntityData(_nextEntityId, tag, entityType));
                _tagToEntityId[tag] = _nextEntityId++;
                entityIds.Add(_tagToEntityId[tag]);
            }
        }

        if (entityIds.Count == 0) return;

        var question = new QuestionData
        {
            Id = int.Parse(element.Attribute("Id")?.Value ?? "0"),
            Title = element.Attribute("Title")?.Value ?? "",
            Body = TruncateBody(element.Attribute("Body")?.Value ?? ""),
            Tags = tags,
            AcceptedAnswerId = int.TryParse(element.Attribute("AcceptedAnswerId")?.Value, out var aid) ? aid : null,
            OwnerUserId = int.Parse(element.Attribute("OwnerUserId")?.Value ?? "-1")
        };
        _questions[question.Id] = question;
        questionCount++;

        if (questionCount % 10000 == 0)
            Console.WriteLine($"    Processed {questionCount} questions...");
    }

    private void ProcessAnswer(XElement element, ref int answerCount)
    {
        var answer = new AnswerData
        {
            Id = int.Parse(element.Attribute("Id")?.Value ?? "0"),
            ParentId = int.Parse(element.Attribute("ParentId")?.Value ?? "0"),
            OwnerUserId = int.Parse(element.Attribute("OwnerUserId")?.Value ?? "-1"),
            Score = int.Parse(element.Attribute("Score")?.Value ?? "0")
        };

        if (_questions.ContainsKey(answer.ParentId))
        {
            var q = _questions[answer.ParentId];
            answer.IsAccepted = q.AcceptedAnswerId == answer.Id;
            _answers.Add(answer);

            // Track user stats
            if (answer.OwnerUserId > 0)
            {
                if (!_userStats.ContainsKey(answer.OwnerUserId))
                    _userStats[answer.OwnerUserId] = new UserStats { UserId = answer.OwnerUserId };

                var stats = _userStats[answer.OwnerUserId];
                stats.AnsweredQuestionIds.Add(answer.ParentId);
                stats.AnswerIds.Add(answer.Id);
                stats.TotalScore += answer.Score;
                foreach (var tag in q.Tags) stats.AnsweredTags.Add(tag);
                foreach (var eid in q.Tags.Where(t => _tagToEntityId.ContainsKey(t))
                                          .Select(t => _tagToEntityId[t]))
                    stats.EntityIds.Add(eid);
            }
            answerCount++;
        }
    }

    private static List<string> ParseTags(string tagString)
    {
        // Handle both formats: <tag1><tag2> and |tag1|tag2|
        return tagString.Split(new[] { '<', '>', '|' }, StringSplitOptions.RemoveEmptyEntries)
                       .Select(t => t.Trim().ToLowerInvariant())
                       .Where(t => !string.IsNullOrWhiteSpace(t))
                       .ToList();
    }

    private static string TruncateBody(string body)
    {
        const int maxLen = 500;
        if (body.Length <= maxLen) return body;
        return body.Substring(0, maxLen) + "...";
    }

    private void BuildKnowledgeGraph()
    {
        Console.WriteLine("  Building tag co-occurrence relations...");
        var cooccurrence = new Dictionary<(int, int), int>();

        foreach (var q in _questions.Values)
        {
            var entityIds = q.Tags.Where(t => _tagToEntityId.ContainsKey(t))
                                  .Select(t => _tagToEntityId[t])
                                  .Distinct().ToList();

            for (int i = 0; i < entityIds.Count; i++)
                for (int j = i + 1; j < entityIds.Count; j++)
                {
                    var key = (Math.Min(entityIds[i], entityIds[j]), Math.Max(entityIds[i], entityIds[j]));
                    cooccurrence[key] = cooccurrence.GetValueOrDefault(key) + 1;
                }
        }

        var topRelations = cooccurrence.OrderByDescending(kv => kv.Value).Take(500);
        foreach (var ((head, tail), _) in topRelations)
            _triples.Add(new TripleData(head, 0, tail));

        Console.WriteLine($"  Created {_triples.Count} knowledge graph triples");
    }

    private async Task SaveOutputAsync()
    {
        Directory.CreateDirectory(_config.OutputPath);
        Directory.CreateDirectory(Path.Combine(_config.OutputPath, "knowledge_graph"));

        var jsonOptions = new JsonSerializerOptions { WriteIndented = true };

        await SaveEntitiesAsync(jsonOptions);
        await SaveRelationsAsync(jsonOptions);
        await SaveTriplesAsync(jsonOptions);
        var experts = await SaveExpertsAsync(jsonOptions);
        var (questionIdMap, _) = await SaveQuestionsAsync(jsonOptions);
        await SaveSamplesAsync(jsonOptions, experts, questionIdMap);
    }

    private async Task SaveEntitiesAsync(JsonSerializerOptions jsonOptions)
    {
        var entitiesPath = Path.Combine(_config.OutputPath, "knowledge_graph", "entities.json");
        await File.WriteAllTextAsync(entitiesPath, JsonSerializer.Serialize(_entities, jsonOptions));
        Console.WriteLine($"  Saved {_entities.Count} entities to entities.json");
    }

    private async Task SaveRelationsAsync(JsonSerializerOptions jsonOptions)
    {
        var relations = new[]
        {
            new RelationData(0, "related_to", "RelatedTo"),
            new RelationData(1, "used_by", "UsedBy"),
            new RelationData(2, "depends_on", "DependsOn"),
            new RelationData(3, "belongs_to", "BelongsTo"),
            new RelationData(4, "uses", "Uses")
        };
        var relationsPath = Path.Combine(_config.OutputPath, "knowledge_graph", "relations.json");
        await File.WriteAllTextAsync(relationsPath, JsonSerializer.Serialize(relations, jsonOptions));
        Console.WriteLine($"  Saved {relations.Length} relations to relations.json");
    }

    private async Task SaveTriplesAsync(JsonSerializerOptions jsonOptions)
    {
        var triplesPath = Path.Combine(_config.OutputPath, "knowledge_graph", "triples.json");
        await File.WriteAllTextAsync(triplesPath, JsonSerializer.Serialize(_triples, jsonOptions));
        Console.WriteLine($"  Saved {_triples.Count} triples to triples.json");
    }

    private async Task<List<ExpertData>> SaveExpertsAsync(JsonSerializerOptions jsonOptions)
    {
        var experts = _userStats.Values
            .Where(u => u.AnswerIds.Count >= _config.MinExpertAnswers)
            .OrderByDescending(u => u.TotalScore)
            .Take(_config.MaxExperts > 0 ? _config.MaxExperts : int.MaxValue)
            .Select((u, idx) => new ExpertData(
                idx,
                $"user_{u.UserId}",
                u.AnsweredTags.Take(10).ToList(),
                u.AnsweredQuestionIds.Take(100).ToList(),
                u.AnswerIds.Take(100).ToList(),
                u.EntityIds.Take(50).ToList()
            ))
            .ToList();

        var expertsPath = Path.Combine(_config.OutputPath, "experts.json");
        await File.WriteAllTextAsync(expertsPath, JsonSerializer.Serialize(experts, jsonOptions));
        Console.WriteLine($"  Saved {experts.Count} experts to experts.json");

        return experts;
    }

    private async Task<(Dictionary<int, int> questionIdMap, List<QuestionOutput> questions)> SaveQuestionsAsync(JsonSerializerOptions jsonOptions)
    {
        var questions = _questions.Values
            .Select((q, idx) => new QuestionOutput(
                idx,
                q.Title,
                q.Body,
                q.Tags,
                q.Tags.Where(t => _tagToEntityId.ContainsKey(t))
                      .Select(t => _tagToEntityId[t]).ToList()
            ))
            .ToList();

        var questionIdMap = _questions.Values.Select((q, idx) => (q.Id, idx)).ToDictionary(x => x.Id, x => x.idx);

        var questionsPath = Path.Combine(_config.OutputPath, "questions.json");
        await File.WriteAllTextAsync(questionsPath, JsonSerializer.Serialize(questions, jsonOptions));
        Console.WriteLine($"  Saved {questions.Count} questions to questions.json");

        return (questionIdMap, questions);
    }

    private async Task SaveSamplesAsync(JsonSerializerOptions jsonOptions, List<ExpertData> experts, Dictionary<int, int> questionIdMap)
    {
        var userToExpertId = experts.ToDictionary(
            e => int.Parse(e.Name.Replace("user_", "")),
            e => e.Id
        );

        var samples = new List<SampleData>();
        var random = new Random(42);

        foreach (var answer in _answers)
        {
            if (!questionIdMap.ContainsKey(answer.ParentId)) continue;
            if (!userToExpertId.ContainsKey(answer.OwnerUserId)) continue;

            var qId = questionIdMap[answer.ParentId];
            var eId = userToExpertId[answer.OwnerUserId];

            // Positive sample (answered)
            samples.Add(new SampleData(qId, eId, true));

            // Negative samples (random experts who didn't answer)
            var otherExperts = experts.Where(e => e.Id != eId).OrderBy(_ => random.Next()).Take(2);
            foreach (var other in otherExperts)
                samples.Add(new SampleData(qId, other.Id, false));
        }

        var samplesPath = Path.Combine(_config.OutputPath, "samples.json");
        await File.WriteAllTextAsync(samplesPath, JsonSerializer.Serialize(samples, jsonOptions));
        Console.WriteLine($"  Saved {samples.Count} training samples to samples.json");
    }
}

