# EPANSERec.DataExtractor

A command-line tool for extracting and processing StackOverflow data dumps into a format suitable for training the EPAN-SERec (Expert Preference Attention Network for Software Expert Recommendation) model.

## Overview

This tool processes StackOverflow XML data dumps and generates:
- **Knowledge Graph**: Entities (tags/technologies), relations, and triples representing technology relationships
- **Expert Profiles**: Users with significant answering history, including their expertise areas
- **Questions**: Software-related questions with associated tags and entities
- **Training Samples**: Positive and negative expert-question pairs for model training

## Prerequisites

- .NET 9.0 SDK
- StackOverflow data dump (XML format)

## Getting the Data

1. Visit [Stack Exchange Data Dump on Archive.org](https://archive.org/download/stackexchange)
2. Download a data dump (recommended smaller datasets for testing):
   - `softwareengineering.stackexchange.com` (~200MB)
   - `codereview.stackexchange.com` (~150MB)
   - `programmers.stackexchange.com` (~100MB)
   - Or the full `stackoverflow.com-Posts.7z` for complete data
3. Extract the `.7z` archive using 7-Zip

## Usage

### Command-Line Mode

```bash
# Basic usage
dotnet run --project EPANSERec.DataExtractor -- -i <path-to-data>

# With all options
dotnet run --project EPANSERec.DataExtractor -- -i C:\data\stackoverflow -o ./output -q 50000 -e 2000 -a 10
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input <path>` | Path to directory containing Posts.xml, Tags.xml | Required |
| `-o, --output <path>` | Output directory | `data/stackoverflow` |
| `-q, --max-questions <n>` | Maximum questions to extract | 10000 |
| `-e, --max-experts <n>` | Maximum experts to include | 1000 |
| `-a, --min-answers <n>` | Minimum answers to qualify as expert | 5 |
| `-h, --help` | Show help message | - |

### Interactive Mode

Run without arguments to enter interactive mode:

```bash
dotnet run --project EPANSERec.DataExtractor
```

Commands:
- `extract <path>` or `e <path>` - Extract data from a StackOverflow dump
- `config` - Show current configuration
- `output <path>` - Set output directory
- `maxq <n>` - Set max questions
- `maxe <n>` - Set max experts
- `mina <n>` - Set min answers for expert qualification
- `download` - Show download instructions
- `help` - Show available commands
- `exit` - Exit the program

## Output Files

The tool generates the following files in the output directory:

```
output/
├── knowledge_graph/
│   ├── entities.json    # Tags as entities with types
│   ├── relations.json   # Relation types (related_to, used_by, etc.)
│   └── triples.json     # Entity-relation-entity triples
├── experts.json         # Expert profiles with expertise tags
├── questions.json       # Questions with tags and entity IDs
└── samples.json         # Training samples (question-expert pairs)
```

## Entity Types

Tags are classified into the following entity types:
- **ProgrammingLanguage**: python, javascript, java, c#, etc.
- **SoftwareFramework**: django, react, angular, spring, etc.
- **SoftwareTool**: mysql, postgresql, mongodb, redis, etc.
- **Tool**: git, docker, kubernetes, aws, etc.
- **Concept**: algorithm, design-patterns, security, etc.

## Using Extracted Data

After extraction, enable real data in the training pipeline:

```csharp
// In EPANSERec.Training/Program.cs
UseRealData = true
```

## Example

```bash
# Extract from a softwareengineering.stackexchange.com dump
dotnet run --project EPANSERec.DataExtractor -- -i "C:\Downloads\softwareengineering.stackexchange.com" -q 20000 -e 500

# Output will be saved to data/stackoverflow/
```

## Performance Notes

The training pipeline uses **multi-threaded parallel processing** for computationally intensive steps:

- **Step 3 (Generate Expertise Preference Graphs)**: Uses `Parallel.ForEach` with 75% of available CPU cores
- Each expert's preference graph is generated independently, making it ideal for parallelization

On a typical 8-core machine, this can provide **4-6x speedup** for Step 3.

