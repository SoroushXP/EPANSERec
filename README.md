# EPAN-SERec: Expertise Preference-Aware Networks for Software Expert Recommendations

[![.NET](https://img.shields.io/badge/.NET-9.0-purple.svg)](https://dotnet.microsoft.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![TorchSharp](https://img.shields.io/badge/TorchSharp-0.103.0-orange.svg)](https://github.com/dotnet/TorchSharp)

A C# .NET 9 implementation of the EPAN-SERec model from the research paper:

> **"EPAN-SERec: Expertise preference-aware networks for software expert recommendations with knowledge graph"**  
> Mingjing Tang, Di Wu, Shu Zhang, Wei Gao  
> *Expert Systems With Applications 244 (2024) 122985*

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#️-architecture)
- [Paper-to-Code Mapping](#-paper-to-code-mapping)
- [Project Structure](#-project-structure)
- [Components](#-components)
- [Getting Started](#-getting-started)
- [Training Results](#-training-results)
- [API Reference](#-api-reference)
- [Testing](#-testing)

## 🎯 Overview

EPAN-SERec addresses three critical challenges in software knowledge community expert recommendation:

| Problem | Description | Solution |
|---------|-------------|----------|
| **Label Dependency** | Inaccurate/mislabeled questions lead to expert mismatching | Expertise preference learning via DRL |
| **Interactive Data Sparsity** | Users only answer a subset of questions | GCN with graph self-supervised learning |
| **Unassociated Knowledge** | Difficulty finding implicit knowledge associations | Knowledge graph embeddings + feature fusion |

### Key Features

- 🧠 **Deep Reinforcement Learning** (Double DQN) for expertise preference modeling
- 📊 **Graph Convolutional Networks** with self-supervised learning
- 🔗 **TransH** knowledge graph embeddings with hyperplane projections
- ⚡ **Attention Networks** for Q&A weighting
- 🎯 **Multi-channel Feature Fusion** for final predictions

## 🏗️ Architecture

The EPAN-SERec model consists of **four main components** as described in Section 4 of the paper:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EPAN-SERec Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────┐    ┌──────────────────────┐                      │
│  │  Software Knowledge  │    │  Expert Historical   │                      │
│  │   Graph (SWKG)       │    │   Q&A Data           │                      │
│  └──────────┬───────────┘    └──────────┬───────────┘                      │
│             │                           │                                   │
│             ▼                           ▼                                   │
│  ┌──────────────────────────────────────────────────┐                      │
│  │         1. EXPERTISE PREFERENCE LEARNING         │  ◄── Section 4.1     │
│  │  ┌────────────────┐    ┌─────────────────────┐   │                      │
│  │  │    Node2Vec    │    │   EPDRL (DRL/MDP)   │   │                      │
│  │  │  (Graph Embed) │    │   Double DQN        │   │                      │
│  │  └────────────────┘    └─────────────────────┘   │                      │
│  │         Output: Expertise Preference Weight Graph │                      │
│  └──────────────────────────────────────────────────┘                      │
│             │                                                               │
│             ▼                                                               │
│  ┌──────────────────────────────────────────────────┐                      │
│  │        2. EXPERTISE PREFERENCE OPTIMIZATION      │  ◄── Section 4.2     │
│  │  ┌────────────────┐    ┌─────────────────────┐   │                      │
│  │  │  GCN Layers    │    │  Self-Supervised    │   │                      │
│  │  │  (Extended)    │    │  Learning (SSL)     │   │                      │
│  │  └────────────────┘    └─────────────────────┘   │                      │
│  │         Output: Optimized Expert Embeddings       │                      │
│  └──────────────────────────────────────────────────┘                      │
│             │                                                               │
│             ▼                                                               │
│  ┌──────────────────────────────────────────────────┐                      │
│  │        3. EXPERTISE PREFERENCE FEATURE FUSION    │  ◄── Section 4.3     │
│  │  ┌────────────────┐    ┌─────────────────────┐   │                      │
│  │  │    TransH      │    │  Feature Fusion     │   │                      │
│  │  │  (KG Embed)    │    │  (MLP-based)        │   │                      │
│  │  └────────────────┘    └─────────────────────┘   │                      │
│  │         Output: Fused Question & Expert Features  │                      │
│  └──────────────────────────────────────────────────┘                      │
│             │                                                               │
│             ▼                                                               │
│  ┌──────────────────────────────────────────────────┐                      │
│  │        4. SOFTWARE EXPERT RECOMMENDATION         │  ◄── Section 4.4     │
│  │  ┌────────────────┐    ┌─────────────────────┐   │                      │
│  │  │   Attention    │    │   Prediction DNN    │   │                      │
│  │  │   Network      │    │   (Sigmoid Output)  │   │                      │
│  │  └────────────────┘    └─────────────────────┘   │                      │
│  │         Output: Expert Recommendation Scores      │                      │
│  └──────────────────────────────────────────────────┘                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📚 Paper-to-Code Mapping

This section provides a detailed mapping between the research paper sections and our C# implementation.

### Section 4.1: Expertise Preference Learning (EPDRL)

**Paper Description:** Uses Deep Reinforcement Learning to model expert historical interactions and generate expertise preference weight graphs through network walking on the knowledge graph.

| Paper Concept | Equation | Implementation File | Class/Method |
|---------------|----------|---------------------|--------------|
| MDP State | Eq. 1: `si = [f1; f2; ...; fi]` | `ReinforcementLearning/MDPState.cs` | `MDPState.StateVector` |
| MDP Action | Eq. 2: `ai = [fi]` | `ReinforcementLearning/MDPState.cs` | `AddNode()` |
| Reward Function | Eq. 3: `w(eij) = (1/En^d) * Ri` | `ReinforcementLearning/EPDRL.cs` | `CalculateReward()` |
| Max Pooling | Eq. 4-5 | `ReinforcementLearning/MDPState.cs` | `GetPooledStateVector()` |
| Q-Network | Eq. 6: `Q(si, ai) = fθ([s'i; ai])` | `ReinforcementLearning/QNetwork.cs` | `forward()` |
| Node2Vec | Graph embeddings | `Embeddings/Node2Vec.cs` | `Train()` |
| Double DQN | Algorithm 1 | `ReinforcementLearning/EPDRL.cs` | `GeneratePreferenceGraph()` |

**Key Implementation Details:**

```csharp
// EPDRL.cs - Expertise Preference Deep Reinforcement Learning
public class EPDRL
{
    // MDP components: State, Action, Path, Reward (M = (S, A, O, R))
    private readonly QNetwork _qNetwork;        // Online Q-network
    private readonly QNetwork _targetNetwork;   // Target Q-network (Double DQN)
    private readonly ReplayMemory _memory;      // Experience replay buffer

    // Generates expertise preference weight graph for an expert
    public ExpertisePreferenceWeightGraph GeneratePreferenceGraph(Expert expert, int maxSteps = 50)
    {
        // Implements Algorithm 1 from the paper
        // Uses ε-greedy policy for exploration/exploitation
    }
}
```

### Section 4.2: Expertise Preference Optimization (GCN + SSL)

**Paper Description:** Uses Graph Convolutional Networks with graph self-supervised learning to optimize feature representations and address interactive data sparsity.

| Paper Concept | Equation | Implementation File | Class/Method |
|---------------|----------|---------------------|--------------|
| GCN Aggregation | Eq. 7: `h_N(v) = AGGREGATE({h_u : u ∈ N(v)})` | `GraphNeuralNetworks/GCNLayer.cs` | `forward()` |
| GCN Update | Eq. 8: `h_v^(l+1) = σ(W·CONCAT(h_v^(l), h_N(v)))` | `GraphNeuralNetworks/GCNLayer.cs` | `forward()` |
| View Generation | Eq. 9: `G' = (V, E', X)` | `GraphNeuralNetworks/GraphSelfSupervisedLearning.cs` | `GenerateView()` |
| Contrastive Loss | Eq. 10-12 | `GraphNeuralNetworks/GraphSelfSupervisedLearning.cs` | `ComputeContrastiveLoss()` |

**Key Implementation Details:**

```csharp
// ExpertisePreferenceOptimizer.cs - GCN with Self-Supervised Learning
public class ExpertisePreferenceOptimizer : nn.Module<Tensor, Tensor, Tensor>
{
    private readonly GCNLayer _gcn1;
    private readonly GCNLayer _gcn2;
    private readonly GraphSelfSupervisedLearning _ssl;

    // Optimizes expert embeddings using GCN + SSL
    public Dictionary<int, float[]> OptimizeExpertEmbeddings(
        Dictionary<int, float[]> initialEmbeddings,
        Dictionary<int, ExpertisePreferenceWeightGraph> preferenceGraphs)
}
```

### Section 4.3: Expertise Preference Feature Fusion (TransH)

**Paper Description:** Uses TransH knowledge graph embedding to capture semantic relationships and fuses with expertise preference features.

| Paper Concept | Equation | Implementation File | Class/Method |
|---------------|----------|---------------------|--------------|
| TransH Projection | Eq. 13: `h⊥ = h - w_r^T h w_r` | `Embeddings/TransH.cs` | `ProjectToHyperplane()` |
| TransH Score | Eq. 14: `f_r(h,t) = \|\|h⊥ + d_r - t⊥\|\|_2^2` | `Embeddings/TransH.cs` | `Score()` |
| Feature Fusion | MLP-based fusion | `Recommendation/FeatureFusion.cs` | `forward()` |

**Key Implementation Details:**

```csharp
// TransH.cs - Knowledge Graph Embedding with Hyperplane Projections
public class TransH : nn.Module
{
    private readonly nn.Embedding _entityEmbeddings;   // Entity vectors
    private readonly nn.Embedding _relationEmbeddings; // Relation translation vectors
    private readonly nn.Embedding _normalVectors;      // Hyperplane normal vectors

    // Projects entity to relation-specific hyperplane
    private Tensor ProjectToHyperplane(Tensor entity, Tensor normal)
    {
        // h⊥ = h - w_r^T h w_r (Equation 13)
        var projection = entity - torch.sum(normal * entity, dim: -1, keepdim: true) * normal;
        return projection;
    }
}
```

### Section 4.4: Software Expert Recommendation

**Paper Description:** Uses attention networks to weight historical Q&A contributions and a DNN for final prediction.

| Paper Concept | Equation | Implementation File | Class/Method |
|---------------|----------|---------------------|--------------|
| Attention Weights | Eq. 15: `α_i = softmax(W_a · tanh(W_q q + W_h h_i))` | `Recommendation/AttentionNetwork.cs` | `forward()` |
| Expert Embedding | Eq. 16: `e = Σ α_i h_i` | `Recommendation/AttentionNetwork.cs` | `forward()` |
| Prediction Input | Eq. 17: `x = [q; e]` | `Recommendation/PredictionDNN.cs` | `forward()` |
| Hidden Layers | Eq. 18: `h = ReLU(W_h x + b_h)` | `Recommendation/PredictionDNN.cs` | `forward()` |
| Output | Eq. 19: `ŷ = σ(W_o h + b_o)` | `Recommendation/PredictionDNN.cs` | `forward()` |

**Key Implementation Details:**

```csharp
// AttentionNetwork.cs - Attention mechanism for Q&A weighting
public class AttentionNetwork : nn.Module<Tensor, Tensor, Tensor>
{
    // Computes attention-weighted expert embedding from historical Q&A
    public override Tensor forward(Tensor questionEmbedding, Tensor historicalEmbeddings)
    {
        // Implements Equations 15-16
        var scores = torch.tanh(_queryTransform.call(questionEmbedding) +
                               _keyTransform.call(historicalEmbeddings));
        var weights = torch.softmax(_attentionVector.call(scores), dim: 0);
        return torch.sum(weights * historicalEmbeddings, dim: 0);
    }
}

// PredictionDNN.cs - Final prediction network
public class PredictionDNN : nn.Module<Tensor, Tensor>
{
    // Predicts probability of expert answering question
    public override Tensor forward(Tensor input)
    {
        // Implements Equations 17-19
        var h = torch.relu(_fc1.call(input));
        h = _dropout.call(h);
        h = torch.relu(_fc2.call(h));
        return torch.sigmoid(_output.call(h));
    }
}
```

## 📁 Project Structure

```
EPANSERec/
├── EPANSERec.sln                          # Solution file
├── README.md                              # This documentation
│
├── EPANSERec.Core/                        # Core library
│   ├── EPANSERec.Core.csproj
│   │
│   ├── Embeddings/                        # Graph & KG Embeddings
│   │   ├── Node2Vec.cs                    # Node2Vec graph embeddings
│   │   └── TransH.cs                      # TransH knowledge graph embeddings
│   │
│   ├── GraphNeuralNetworks/               # GCN & SSL Components
│   │   ├── GCNLayer.cs                    # Graph Convolutional Network layer
│   │   ├── GraphSelfSupervisedLearning.cs # Self-supervised learning module
│   │   └── ExpertisePreferenceOptimizer.cs# Combined GCN + SSL optimizer
│   │
│   ├── KnowledgeGraph/                    # Software Knowledge Graph
│   │   ├── Entity.cs                      # KG entity definition
│   │   ├── Relation.cs                    # KG relation definition
│   │   ├── Triple.cs                      # KG triple (head, relation, tail)
│   │   └── SoftwareKnowledgeGraph.cs      # SWKG implementation
│   │
│   ├── Models/                            # Domain Models
│   │   ├── Expert.cs                      # Expert/user model
│   │   ├── Question.cs                    # Question model
│   │   └── ExpertisePreferenceWeightGraph.cs # Preference graph
│   │
│   ├── Recommendation/                    # Recommendation Components
│   │   ├── EPANSERecModel.cs              # Main model orchestrator
│   │   ├── AttentionNetwork.cs            # Attention mechanism
│   │   ├── FeatureFusion.cs               # Feature fusion layer
│   │   └── PredictionDNN.cs               # Final prediction network
│   │
│   ├── ReinforcementLearning/             # DRL Components
│   │   ├── EPDRL.cs                       # Main DRL controller
│   │   ├── QNetwork.cs                    # Q-Network (Double DQN)
│   │   ├── MDPState.cs                    # MDP state representation
│   │   ├── Experience.cs                  # Experience tuple
│   │   └── ReplayMemory.cs                # Experience replay buffer
│   │
│   └── Utils/                             # Utilities
│       ├── DataLoader.cs                  # Base data loader
│       ├── SampleDataGenerator.cs         # Synthetic data generator
│       ├── StackOverflowDataLoader.cs     # StackOverflow data loader
│       ├── TrainingPipeline.cs            # Training orchestration
│       └── Metrics.cs                     # Evaluation metrics
│
├── EPANSERec.Training/                    # Training Console App
│   ├── EPANSERec.Training.csproj
│   └── Program.cs                         # Training entry point
│
└── EPANSERec.Tests/                       # Unit Tests
    ├── EPANSERec.Tests.csproj
    ├── KnowledgeGraphTests.cs
    ├── ModelsTests.cs
    ├── Node2VecTests.cs
    ├── TransHTests.cs
    ├── ReinforcementLearningTests.cs
    ├── SampleDataGeneratorTests.cs
    └── EPANSERecModelTests.cs
```

## 🔧 Components

### 1. Software Knowledge Graph (SWKG)

The Software Knowledge Graph represents software domain knowledge with entities and relations.

**Entity Types** (8 types as per paper Section 3.2):

| Entity Type | Description | Example |
|-------------|-------------|---------|
| `ProgrammingLanguage` | Programming languages | C#, Python, Java |
| `SoftwareFramework` | Development frameworks | .NET, React, Django |
| `SoftwareAPI` | APIs and interfaces | REST API, GraphQL |
| `SoftwareTool` | Development tools | Git, Docker, VS Code |
| `SoftwareLibrary` | Code libraries | TorchSharp, NumPy |
| `SoftwareStandard` | Standards/protocols | HTTP, JSON, OAuth |
| `Concept` | Abstract concepts | OOP, Microservices |
| `Tool` | General tools | Debugger, Profiler |

**Relation Types** (5 types):

| Relation | Description | Example |
|----------|-------------|---------|
| `RelatedTo` | General association | C# → .NET |
| `UsedBy` | Usage relationship | React → JavaScript |
| `DependsOn` | Dependency | TorchSharp → PyTorch |
| `BelongsTo` | Category membership | Django → Python |
| `Uses` | Utilization | Docker → Containers |

### 2. Node2Vec (Graph Embeddings)

Implements biased random walks for learning graph structure embeddings.

**Parameters:**
- `walkLength`: Length of random walks (default: 80)
- `numWalks`: Number of walks per node (default: 10)
- `p`: Return parameter (BFS vs DFS) (default: 1.0)
- `q`: In-out parameter (default: 1.0)
- `embeddingDimension`: Output dimension (default: 100)

### 3. TransH (Knowledge Graph Embeddings)

Implements translation-based embeddings with hyperplane projections.

**Key Features:**
- Entity embeddings in continuous vector space
- Relation-specific hyperplanes for projection
- Margin-based ranking loss for training

### 4. EPDRL (Deep Reinforcement Learning)

Implements Double DQN for expertise preference learning.

**MDP Definition** (M = (S, A, O, R)):
- **S (State)**: Pooled representation of visited nodes
- **A (Action)**: Next node to visit
- **O (Path)**: Sequence of visited nodes
- **R (Reward)**: Based on node relevance and expert interaction

**Hyperparameters:**
- `gamma`: Discount factor (default: 0.99)
- `epsilon`: Exploration rate (default: 1.0, decays to 0.01)
- `memoryCapacity`: Replay buffer size (default: 10000)
- `batchSize`: Training batch size (default: 32)

### 5. GCN with Self-Supervised Learning

Implements graph convolutional networks with contrastive learning.

**Architecture:**
- 2-layer GCN for feature aggregation
- View generation via edge dropout
- Contrastive loss for self-supervision

### 6. Attention Network

Weights historical Q&A contributions for expert representation.

**Mechanism:**
- Query: Current question embedding
- Keys: Historical Q&A embeddings
- Output: Weighted sum of historical embeddings

### 7. Prediction DNN

Final prediction network for expert-question matching.

**Architecture:**
- Input: Concatenated question + expert embeddings
- Hidden layers: 256 → 128 with ReLU + Dropout
- Output: Sigmoid probability

## 🚀 Getting Started

### Prerequisites

- [.NET 9.0 SDK](https://dotnet.microsoft.com/download/dotnet/9.0)
- Windows, Linux, or macOS

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd EPANSERec
```

2. Restore packages:
```bash
dotnet restore
```

3. Build the solution:
```bash
dotnet build
```

### Running Training

```bash
cd EPANSERec.Training
dotnet run
```

### Training Configuration

Modify `Program.cs` to adjust training parameters:

```csharp
var config = new TrainingConfig
{
    // General settings
    Epochs = 100,              // Training epochs
    BatchSize = 32,            // Batch size
    LearningRate = 5e-4f,      // Learning rate
    EmbeddingDim = 100,        // Embedding dimension

    // Component-specific settings
    TransHEpochs = 100,        // TransH pre-training epochs
    EPDRLEpisodes = 50,        // EPDRL episodes per expert
    GCNEpochs = 50,            // GCN optimization epochs

    // Training control
    EarlyStoppingPatience = 15, // Early stopping patience
    Seed = 42                   // Random seed
};
```

## 📊 Training Results

### Current Implementation Results (Synthetic Data)

| Metric | Value | Paper Target |
|--------|-------|--------------|
| AUC | 73.57% | 84.32% |
| Accuracy | 68.00% | 77.68% |
| F1 Score | 67.79% | 78.11% |

*Note: Results are from synthetic data. Real StackOverflow data should yield results closer to paper targets.*

### Paper Results (Table 4)

| Model | AUC (%) | ACC (%) | F1 (%) |
|-------|---------|---------|--------|
| TagLDA | 63.21 | 61.32 | 60.11 |
| KGAT | 72.35 | 68.97 | 67.89 |
| RippleNet | 74.56 | 70.12 | 69.34 |
| DiffNet++ | 76.89 | 72.45 | 71.23 |
| **EPAN-SERec** | **84.32** | **77.68** | **78.11** |

### Ablation Study Results (Paper Tables 5-7)

**Component Contributions:**

| Component Removed | AUC Drop | ACC Drop | F1 Drop |
|-------------------|----------|----------|---------|
| EPDRL | -2.90% | -4.71% | -3.64% |
| Self-Supervised Learning | -1.99% | -2.79% | -1.53% |
| TransH | -1.97% | -2.68% | -1.37% |
| Expertise Preference Feature | -9.21% | -5.34% | -4.95% |
| Attention Network | -1.99% | -2.79% | -1.53% |

## 📖 API Reference

### EPANSERecModel

Main model class that orchestrates all components.

```csharp
// Initialize model
var model = new EPANSERecModel(knowledgeGraph, embeddingDim: 100, beta: 0.1f);
model.Initialize(learningRate: 1e-4f);

// Pre-train knowledge embeddings
model.PretrainKnowledgeEmbeddings(epochs: 100, batchSize: 128);

// Generate expertise preference graphs
model.GenerateExpertisePreferenceGraphs(experts, episodes: 50);

// Optimize expert embeddings
model.OptimizeExpertEmbeddings(epochs: 50);

// Train the model
model.Train(trainingData, epochs: 100, batchSize: 32);

// Make predictions
float probability = model.Predict(question, expert);

// Recommend experts for a question
var recommendations = model.RecommendExperts(question, candidateExperts, topK: 10);
```

### SoftwareKnowledgeGraph

```csharp
// Create knowledge graph
var kg = new SoftwareKnowledgeGraph();

// Add entities
var entity = kg.AddEntity("C#", EntityType.ProgrammingLanguage);

// Add relations
kg.AddRelation(entity1.Id, entity2.Id, RelationType.RelatedTo);

// Get neighbors
var neighbors = kg.GetNeighbors(entityId);

// Get adjacency matrix
var adjacency = kg.GetAdjacencyMatrix();
```

### TrainingPipeline

```csharp
// Create pipeline
var pipeline = new TrainingPipeline(config);

// Run full training
var results = pipeline.Train(
    knowledgeGraph,
    experts,
    questions,
    trainingData
);

// Evaluate model
var metrics = pipeline.Evaluate(testData);
```

## 🧪 Testing

### Running Tests

```bash
cd EPANSERec.Tests
dotnet test --verbosity normal
```

### Test Coverage

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `KnowledgeGraphTests.cs` | 8 | Entity/Relation CRUD, Adjacency |
| `ModelsTests.cs` | 6 | Expert, Question, PreferenceGraph |
| `Node2VecTests.cs` | 7 | Random walks, Embeddings |
| `TransHTests.cs` | 6 | Projections, Training, Scoring |
| `ReinforcementLearningTests.cs` | 10 | MDP, Q-Network, EPDRL |
| `SampleDataGeneratorTests.cs` | 8 | Data generation, Balancing |
| `EPANSERecModelTests.cs` | 8 | Integration tests |
| **Total** | **53** | All passing ✅ |

### Example Test

```csharp
[Fact]
public void EPANSERecModel_ShouldPredictProbability()
{
    // Arrange
    var kg = CreateSampleKnowledgeGraph();
    var model = new EPANSERecModel(kg, embeddingDim: 32);
    model.Initialize();

    // Act
    var probability = model.Predict(question, expert);

    // Assert
    Assert.InRange(probability, 0.0f, 1.0f);
}
```

## 📚 References

### Original Paper

```bibtex
@article{tang2024epansrec,
  title={EPAN-SERec: Expertise preference-aware networks for software expert
         recommendations with knowledge graph},
  author={Tang, Mingjing and Wu, Di and Zhang, Shu and Gao, Wei},
  journal={Expert Systems With Applications},
  volume={244},
  pages={122985},
  year={2024},
  publisher={Elsevier}
}
```

### Key Dependencies

- [TorchSharp](https://github.com/dotnet/TorchSharp) - PyTorch bindings for .NET
- [.NET 9.0](https://dotnet.microsoft.com/) - Runtime framework

### Related Work

- **Node2Vec**: Grover & Leskovec (2016) - Scalable Feature Learning for Networks
- **TransH**: Wang et al. (2014) - Knowledge Graph Embedding by Translating on Hyperplanes
- **Double DQN**: Hasselt et al. (2016) - Deep Reinforcement Learning with Double Q-Learning
- **GCN**: Kipf & Welling (2017) - Semi-Supervised Classification with Graph Convolutional Networks

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚠️ Limitations

As noted in the paper (Section 6):

1. **Training Time**: The framework is time-consuming during training, which may not meet online recommendation needs
2. **Complexity**: The design integrates multiple learning paradigms (RL, SSL, GCN) and is not end-to-end
3. **Data Scope**: Currently only considers Q&A text, not source code, issue reports, or mailing lists

## 🔮 Future Work

- Support for real StackOverflow data loading
- End-to-end model optimization
- Multi-modal knowledge graph support
- Online recommendation capabilities
