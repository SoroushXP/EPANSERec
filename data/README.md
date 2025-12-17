# StackOverflow Dataset Format

This directory contains sample data files showing the expected format for the EPAN-SERec model.
Replace these with real StackOverflow data for production use.

## Directory Structure

```
data/stackoverflow/
├── knowledge_graph/
│   ├── entities.json       # Software entities (tags, technologies)
│   ├── relations.json      # Relation type definitions
│   └── triples.json        # Entity-relation-entity triples
├── experts.json            # User/expert data
├── questions.json          # Question data
└── samples.json            # Training samples (question, expert, answered)
```

## File Formats

### entities.json
Software entities representing technologies, frameworks, concepts, etc.

```json
[
  {"id": 1, "name": "python", "type": "ProgrammingLanguage"},
  {"id": 2, "name": "django", "type": "SoftwareFramework"},
  {"id": 3, "name": "rest-api", "type": "Concept"}
]
```

**Entity Types:**
- `ProgrammingLanguage` - Programming languages (Python, Java, C#, etc.)
- `SoftwareFramework` - Frameworks (Django, React, Spring, etc.)
- `SoftwareAPI` - APIs and interfaces
- `SoftwareTool` - Development tools
- `SoftwareLibrary` - Code libraries
- `SoftwareStandard` - Standards and protocols
- `Concept` - Abstract concepts (OOP, REST, etc.)
- `Tool` - General tools

### relations.json
Relation types for the knowledge graph.

```json
[
  {"id": 1, "name": "related_to", "type": "RelatedTo"},
  {"id": 2, "name": "used_by", "type": "UsedBy"}
]
```

**Relation Types:**
- `RelatedTo` - General association
- `UsedBy` - Usage relationship
- `DependsOn` - Dependency relationship
- `BelongsTo` - Category membership
- `Uses` - Utilization relationship

### triples.json
Knowledge graph triples (head_entity, relation, tail_entity).

```json
[
  {"headId": 2, "relationId": 4, "tailId": 1}
]
```
This means: Django (id=2) BelongsTo (id=4) Python (id=1)

### experts.json
User/expert information with their expertise and history.

```json
[
  {
    "id": 1,
    "name": "python_expert",
    "expertiseTags": ["python", "django"],
    "historicalQuestionIds": [1, 2, 3],
    "historicalAnswerIds": [101, 102],
    "historicalEntityIds": [1, 5, 9]
  }
]
```

### questions.json
Question data with tags and related entities.

```json
[
  {
    "id": 1,
    "title": "How to create a REST API in Django?",
    "body": "I'm trying to build a REST API...",
    "tags": ["python", "django", "rest-api"],
    "entityIds": [1, 5, 9]
  }
]
```

### samples.json
Training samples indicating which experts answered which questions.

```json
[
  {"questionId": 1, "expertId": 1, "answered": true},
  {"questionId": 1, "expertId": 2, "answered": false}
]
```

## Usage

To use real StackOverflow data, set `UseRealData = true` in the configuration:

```csharp
var config = new TrainingConfig
{
    UseRealData = true,
    DataPath = "data/stackoverflow",  // Path to your data
    // ... other settings
};
```

## Preparing Real StackOverflow Data

To prepare real StackOverflow data:

1. **Download** the StackOverflow data dump from https://archive.org/details/stackexchange
2. **Extract** relevant tags as entities
3. **Build** the knowledge graph from tag co-occurrences
4. **Export** user expertise from answer history
5. **Generate** training samples from Q&A pairs

The larger and more complete your dataset, the better the model will perform.

