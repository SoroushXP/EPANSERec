using EPANSERec.Core.KnowledgeGraph;
using EPANSERec.Core.Models;

namespace EPANSERec.Core.Utils;

/// <summary>
/// Generates sample StackOverflow-like data for testing the EPAN-SERec model.
/// </summary>
public class SampleDataGenerator
{
    private readonly Random _random;
    private readonly string[] _programmingLanguages = { "C#", "Python", "Java", "JavaScript", "TypeScript", "Go", "Rust" };
    private readonly string[] _frameworks = { "ASP.NET", "Django", "Spring", "React", "Angular", "Vue", "Node.js" };
    private readonly string[] _concepts = { "OOP", "Async", "REST", "GraphQL", "Database", "Testing", "Security" };
    private readonly string[] _tools = { "Git", "Docker", "Kubernetes", "Azure", "AWS", "CI/CD", "VS Code" };

    public SampleDataGenerator(int? seed = null)
    {
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Generates a complete sample dataset.
    /// </summary>
    public (SoftwareKnowledgeGraph kg, List<Expert> experts, List<Question> questions, 
        List<(int questionId, int expertId, bool answered)> samples) GenerateDataset(
        int numExperts = 100, int numQuestions = 500, int numSamples = 2000)
    {
        var kg = GenerateKnowledgeGraph();
        var experts = GenerateExperts(numExperts, kg);
        var questions = GenerateQuestions(numQuestions, kg);
        var samples = GenerateTrainingSamples(questions, experts, numSamples);
        
        return (kg, experts, questions, samples);
    }

    /// <summary>
    /// Generates a software knowledge graph.
    /// </summary>
    public SoftwareKnowledgeGraph GenerateKnowledgeGraph()
    {
        var kg = new SoftwareKnowledgeGraph();
        int entityId = 0;
        
        // Add programming languages
        foreach (var lang in _programmingLanguages)
            kg.AddEntity(new Entity(entityId++, lang, EntityType.ProgrammingLanguage));

        // Add frameworks
        foreach (var fw in _frameworks)
            kg.AddEntity(new Entity(entityId++, fw, EntityType.SoftwareFramework));

        // Add concepts
        foreach (var concept in _concepts)
            kg.AddEntity(new Entity(entityId++, concept, EntityType.SoftwareStandard));

        // Add tools
        foreach (var tool in _tools)
            kg.AddEntity(new Entity(entityId++, tool, EntityType.SoftwareTool));

        // Add relations
        kg.AddRelation(new Relation(0, RelationType.UsedBy, "uses"));
        kg.AddRelation(new Relation(1, RelationType.RelatedTo, "related_to"));
        kg.AddRelation(new Relation(2, RelationType.PartOf, "belongs_to"));
        
        // Add triples (relationships between entities)
        var entities = kg.Entities.Values.ToList();
        for (int i = 0; i < entities.Count; i++)
        {
            // Create some relationships
            int numRelations = _random.Next(2, 5);
            for (int j = 0; j < numRelations; j++)
            {
                int targetIdx = _random.Next(entities.Count);
                if (targetIdx != i)
                {
                    int relationId = _random.Next(3);
                    kg.AddTriple(entities[i].Id, relationId, entities[targetIdx].Id);
                }
            }
        }
        
        return kg;
    }

    /// <summary>
    /// Generates sample experts.
    /// </summary>
    public List<Expert> GenerateExperts(int count, SoftwareKnowledgeGraph kg)
    {
        var experts = new List<Expert>();
        var entityIds = kg.Entities.Keys.ToList();
        
        for (int i = 0; i < count; i++)
        {
            var expert = new Expert(i, $"Expert_{i}");
            
            // Assign expertise tags
            int numTags = _random.Next(2, 6);
            var allTags = _programmingLanguages.Concat(_frameworks).Concat(_concepts).ToList();
            for (int j = 0; j < numTags; j++)
                expert.ExpertiseTags.Add(allTags[_random.Next(allTags.Count)]);
            
            // Assign historical entities (questions they've answered)
            int numHistorical = _random.Next(5, 20);
            for (int j = 0; j < numHistorical; j++)
                expert.HistoricalEntityIds.Add(entityIds[_random.Next(entityIds.Count)]);
            
            experts.Add(expert);
        }
        
        return experts;
    }

    /// <summary>
    /// Generates sample questions.
    /// </summary>
    public List<Question> GenerateQuestions(int count, SoftwareKnowledgeGraph kg)
    {
        var questions = new List<Question>();
        var entityIds = kg.Entities.Keys.ToList();
        
        for (int i = 0; i < count; i++)
        {
            var question = new Question(i, $"Question about topic {i}", $"Body of question {i}");
            
            // Assign tags
            int numTags = _random.Next(1, 4);
            var allTags = _programmingLanguages.Concat(_frameworks).Concat(_concepts).ToList();
            for (int j = 0; j < numTags; j++)
                question.Tags.Add(allTags[_random.Next(allTags.Count)]);
            
            // Assign entity IDs
            int numEntities = _random.Next(1, 5);
            for (int j = 0; j < numEntities; j++)
                question.EntityIds.Add(entityIds[_random.Next(entityIds.Count)]);
            
            questions.Add(question);
        }
        
        return questions;
    }

    /// <summary>
    /// Generates training samples (question-expert pairs with labels).
    /// Ensures balanced positive/negative samples for better training.
    /// </summary>
    public List<(int questionId, int expertId, bool answered)> GenerateTrainingSamples(
        List<Question> questions, List<Expert> experts, int count)
    {
        var samples = new List<(int, int, bool)>();
        var usedPairs = new HashSet<(int, int)>();

        int targetPositives = count / 2;  // Aim for 50% positive samples
        int positiveCount = 0;
        int negativeCount = 0;

        // First, generate positive samples based on tag overlap
        foreach (var question in questions)
        {
            if (positiveCount >= targetPositives) break;

            // Find experts with matching tags
            var matchingExperts = experts
                .Where(e => question.Tags.Intersect(e.ExpertiseTags).Any())
                .OrderByDescending(e => question.Tags.Intersect(e.ExpertiseTags).Count())
                .Take(5)
                .ToList();

            foreach (var expert in matchingExperts)
            {
                if (positiveCount >= targetPositives) break;
                var pair = (question.Id, expert.Id);
                if (!usedPairs.Contains(pair))
                {
                    usedPairs.Add(pair);
                    samples.Add((question.Id, expert.Id, true));
                    positiveCount++;

                    // Record this in expert's history
                    expert.HistoricalQuestionIds.Add(question.Id);
                }
            }
        }

        // Then, generate negative samples (no tag overlap or random non-matching)
        int targetNegatives = count - positiveCount;
        while (negativeCount < targetNegatives)
        {
            var question = questions[_random.Next(questions.Count)];
            var expert = experts[_random.Next(experts.Count)];
            var pair = (question.Id, expert.Id);

            if (!usedPairs.Contains(pair))
            {
                // Prefer experts with no tag overlap for negative samples
                var tagOverlap = question.Tags.Intersect(expert.ExpertiseTags).Count();
                if (tagOverlap == 0 || _random.NextDouble() < 0.3)
                {
                    usedPairs.Add(pair);
                    samples.Add((question.Id, expert.Id, false));
                    negativeCount++;
                }
            }
        }

        // Shuffle samples
        return samples.OrderBy(_ => _random.Next()).ToList();
    }
}

