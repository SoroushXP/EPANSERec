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
| Reward Function | Eq. 3: `w(eij) = (1/e^d) * Ri` | `ReinforcementLearning/EPDRL.cs` | `AssignPathWeights()` |
| State Concatenation | Eq. 5: `O_l = [f1; f2; ...; f(i-1)]` | `ReinforcementLearning/MDPState.cs` | `StateVector` |
| Mean Pooling | Eq. 6: `O'_l = sum(O_l) / n` | `ReinforcementLearning/MDPState.cs` | `GetPooledStateVector()` |
| Q-Network | Eq. 7: `Q(si, ai) = fθ([s'i; ai])` | `ReinforcementLearning/QNetwork.cs` | `forward()` |
| Node2Vec | Graph embeddings | `Embeddings/Node2Vec.cs` | `Train()` |
| Double DQN | Algorithm 1 | `ReinforcementLearning/EPDRL.cs` | `TrainOnBatch()` |

**Key Implementation Details:**

```csharp
// EPDRL.cs - Expertise Preference Deep Reinforcement Learning
public class EPDRL
{
    // MDP components: State, Action, Path, Reward (M = (S, A, O, R))
    private readonly QNetwork _policyNetwork;   // Policy Q-network (action selection)
    private readonly QNetwork _targetNetwork;   // Target Q-network (action evaluation)
    private readonly ReplayMemory _memory;      // Experience replay buffer

    // Double DQN Training: Q_target = r + γ * Q_target(s', argmax_a Q_policy(s', a))
    private void TrainOnBatch()
    {
        // Step 1: Use POLICY network to select best action (argmax)
        // Step 2: Use TARGET network to evaluate that action
        // This reduces overestimation bias compared to standard DQN
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

**Paper Description:** Uses TransH knowledge graph embedding to capture semantic relationships and fuses with expertise preference features using a gating mechanism.

| Paper Concept | Equation | Implementation File | Class/Method |
|---------------|----------|---------------------|--------------|
| TransH Projection | Eq. 13: `h⊥ = h - w_r^T h w_r` | `Embeddings/TransH.cs` | `ProjectToHyperplane()` |
| TransH Score | Eq. 14: `f_r(h,t) = \|\|h⊥ + d_r - t⊥\|\|_2^2` | `Embeddings/TransH.cs` | `ScoreFunction()` |
| Gated Feature Fusion | `g ⊙ e_pref + (1-g) ⊙ e_sem` | `Recommendation/AttentionNetwork.cs` | `FeatureFusion.forward()` |

**Key Implementation Details:**

```csharp
// TransH.cs - Knowledge Graph Embedding with Hyperplane Projections
public class TransH : Module
{
    private readonly Embedding _entityEmbeddings;   // Entity vectors
    private readonly Embedding _relationEmbeddings; // Relation translation vectors
    private readonly Embedding _normalVectors;      // Hyperplane normal vectors

    // Projects entity to relation-specific hyperplane (Equation 13)
    private Tensor ProjectToHyperplane(Tensor entityEmb, Tensor normalVector)
    {
        // h⊥ = h - (h · w_r) * w_r
        var dotProduct = (entityEmb * normalVector).sum(dim: -1, keepdim: true);
        return entityEmb - dotProduct * normalVector;
    }
}

// FeatureFusion.cs - Gating mechanism for adaptive feature fusion
public class FeatureFusion : Module<Tensor, Tensor, Tensor>
{
    // g = σ(W_g · [e_pref; e_sem] + b_g)
    // e_fused = g ⊙ e_pref + (1-g) ⊙ e_sem
    public override Tensor forward(Tensor expertisePreferenceEmb, Tensor semanticEmb)
    {
        var gate = functional.sigmoid(_gateLayer.forward(concatenated));
        return gate * prefTransformed + (1 - gate) * semTransformed;
    }
}
```

### Section 4.4: Software Expert Recommendation

**Paper Description:** Uses attention networks to weight historical Q&A contributions and a DNN for final prediction. Joint training with BCE + SSL loss (Equation 19).

| Paper Concept | Equation | Implementation File | Class/Method |
|---------------|----------|---------------------|--------------|
| Question Self-Attention | Scaled dot-product | `Recommendation/AttentionNetwork.cs` | `QuestionAttentionNetwork.forward()` |
| Expert Attention Weights | Eq. 15: `α_k = softmax(e(q') · e(q_k))` | `Recommendation/AttentionNetwork.cs` | `AttentionNetwork.forward()` |
| Expert Embedding | Eq. 16: `e(ui) = Σ α_k · e(q_k)` | `Recommendation/AttentionNetwork.cs` | `AttentionNetwork.forward()` |
| Prediction Input | Eq. 17: `x = [e(q'); e(ui)]` | `Recommendation/PredictionDNN.cs` | `forward()` |
| BCE Loss | Eq. 18: `L(θ) = -Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]` | `Recommendation/PredictionDNN.cs` | `RecommendationLoss.ComputeBCELoss()` |
| Joint Loss | Eq. 19: `L = L(θ) + β·L_con` | `Recommendation/EPANSERecModel.cs` | `TrainBatch()` |

**Key Implementation Details:**

```csharp
// QuestionAttentionNetwork.cs - Self-attention for question embeddings
public class QuestionAttentionNetwork : Module<Tensor, Tensor>
{
    // Scaled dot-product self-attention: Attention(Q,K,V) = softmax(QK^T/√d)V
    public override Tensor forward(Tensor entityEmbeddings)
    {
        var queries = _queryLayer.forward(entityEmbeddings);
        var keys = _keyLayer.forward(entityEmbeddings);
        var values = _valueLayer.forward(entityEmbeddings);
        var attentionScores = torch.matmul(queries, keys.transpose(0, 1)) / _scale;
        var attended = torch.matmul(functional.softmax(attentionScores, dim: -1), values);
        return attended.mean(dimensions: new long[] { 0 });
    }
}

// AttentionNetwork.cs - Attention for expert historical Q&A
public class AttentionNetwork : Module<Tensor, Tensor, Tensor>
{
    // Computes attention-weighted expert embedding (Equations 15-16)
    public override Tensor forward(Tensor questionEmbedding, Tensor historicalEmbeddings)
    {
        var dotProducts = torch.matmul(historicalEmbeddings, questionEmbedding.unsqueeze(0).transpose(0, 1));
        var attentionWeights = functional.softmax(dotProducts, dim: 0);
        return (historicalEmbeddings * attentionWeights).sum(dim: 0);
    }
}

// EPANSERecModel.cs - Joint BCE + SSL Training (Equation 19)
public float TrainBatch(List<(Question, Expert, bool)> batch)
{
    var bceLoss = RecommendationLoss.ComputeBCELoss(predictions, labels);
    var sslLoss = ComputeBatchSSLLoss(expertIdsInBatch);  // L_con
    var totalLoss = bceLoss + sslLoss;  // L = L(θ) + β * L_con
    totalLoss.backward();
    return totalLoss.item<float>();
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

Implements biased random walks for learning graph structure embeddings, based on Grover & Leskovec (2016).

**Key Features:**
- Biased random walks with configurable BFS/DFS exploration via `p` and `q` parameters
- Skip-gram with Negative Sampling (SGNS) using proper **unigram^0.75 distribution**
- Linear learning rate decay during training
- Dynamic context window sampling

**Parameters:**
- `walkLength`: Length of random walks (default: 80, paper recommends 40-80)
- `numWalks`: Number of walks per node (default: 10)
- `p`: Return parameter - high p (>1) = less likely to return (default: 1.0)
- `q`: In-out parameter - high q (>1) = BFS-like, low q (<1) = DFS-like (default: 1.0)
- `embeddingDimension`: Output dimension (default: 100)
- `negSamples`: Negative samples per positive (default: 5, Word2Vec recommends 5-20)

### 3. TransH (Knowledge Graph Embeddings)

Implements translation-based embeddings with hyperplane projections.

**Key Features:**
- Entity embeddings in continuous vector space
- Relation-specific hyperplanes for projection
- Margin-based ranking loss for training

### 4. EPDRL (Deep Reinforcement Learning)

Implements Double DQN for expertise preference learning (Paper Section 4.1).

**MDP Definition** (M = (S, A, O, R)):
- **S (State)**: `s_i = [O'_l; f_i]` - pooled history + current node embedding (Eq. 5-6)
- **A (Action)**: Next node to visit (represented by node embedding)
- **O (Path)**: Sequence of visited nodes forming expertise preference path
- **R (Reward)**: Shaped reward function:
  - `+R_pos` (+1.0): Reaching target entity (expert's historical interaction)
  - `-R_neg` (-1.0): Max path length reached without success
  - `-R_step` (-0.01): Per-step cost to encourage shorter paths

**Q-Network Architecture:**
- 4-layer MLP: input → 256 → 256 → 128 → 1
- ReLU activations with light dropout (0.1)
- Follows standard DQN architecture from Mnih et al. (2015)

**Hyperparameters:**
- `gamma`: Discount factor (default: 0.99, standard DRL value)
- `epsilon`: Exploration rate (default: 1.0, decays to 0.01)
- `memoryCapacity`: Replay buffer size (default: 10000)
- `batchSize`: Training batch size (default: 128)

### 5. GCN with Self-Supervised Learning

Implements graph convolutional networks with **mutual information maximization** (Paper Section 4.2, Eq. 11-12).

**Architecture:**
- 2-layer GCN for feature aggregation
- **Global MI** (default): Between node embeddings and full graph summary
- **Hierarchical MI** (optional, for large datasets): Adds subgraph-level MI
- Positive/negative sampling via node shuffling (DGI-style)

**Loss Function (Eq. 12):**
- `L_con = -E[log(fD(h_i, s_G))] - E[log(1 - fD(h̃_j, s_G))]`
- With hierarchical MI enabled: `L_ssl = L_global + α × L_hierarchical`
- γ coefficient reduces gradient conflict with main task

**Configuration:**
- `useHierarchicalMI`: Enable hierarchical MI (default: `false` for small datasets, enable for large real-world data)

### 6. Attention Networks

Two attention mechanisms for question and expert representations.

**Question Self-Attention (QuestionAttentionNetwork):**
- Scaled dot-product self-attention over entity embeddings
- Attention(Q,K,V) = softmax(QK^T/√d)V
- Outputs rich question representation

**Expert Attention (AttentionNetwork):**
- Query: Current question embedding
- Keys: Historical Q&A embeddings
- Output: Attention-weighted sum of historical embeddings

### 7. Feature Fusion (Gating)

Adaptive fusion of expertise preference and semantic embeddings.

**Gating Mechanism:**
- Gate: `g = σ(W_g · [e_pref; e_sem] + b_g)`
- Fusion: `e_fused = g ⊙ e_pref + (1-g) ⊙ e_sem`
- Learns to balance contributions from each source

### 8. Prediction DNN

Final prediction network for expert-question matching with joint loss.

**Architecture:**
- Input: Concatenated question + expert embeddings
- Hidden layers: 256 → 128 → 64 with ReLU + BatchNorm + Dropout
- Output: Sigmoid probability
- **Joint Loss**: L = BCE + β × SSL (Equation 19)

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
    BatchSize = 128,           // Batch size (paper default)
    LearningRate = 1e-4f,      // Learning rate (Adam optimizer)
    EmbeddingDim = 100,        // Embedding dimension (paper default, range: 50-200)

    // Component-specific settings
    TransHEpochs = 100,        // TransH pre-training epochs
    EPDRLEpisodes = 50,        // EPDRL episodes per expert
    GCNEpochs = 50,            // GCN optimization epochs

    // Training control
    EarlyStoppingPatience = 15, // Early stopping patience
    Seed = 42                   // Random seed
};
```

### Hyperparameters Reference (Paper Section 5.2 / Table 3)

The following hyperparameters are used in the paper's experiments on StackOverflow data:

| Component | Parameter | Default Value | Description |
|-----------|-----------|---------------|-------------|
| **General** | Embedding dim | 100 | Entity/node embedding dimension (range: 50-200) |
| | Learning rate | 1e-4 | Adam optimizer learning rate |
| | Batch size | 128 | Training batch size |
| **EPDRL** | γ (gamma) | 0.99 | Discount factor for future rewards |
| | ε (epsilon) | 1.0 → 0.01 | Exploration rate with decay |
| | Max path length | 10 | Maximum steps per episode |
| | Replay buffer | 10,000 | Experience replay capacity |
| | R_pos | +1.0 | Reward for reaching target |
| | R_step | -0.01 | Per-step cost |
| **TransH** | Margin (γ) | 1.0 | Margin for ranking loss |
| **SSL** | β (beta) | 0.1 | SSL loss coefficient (Eq. 19) |
| | γ (gamma) | 0.1 | Gradient conflict reduction |
| **Node2Vec** | Walk length | 80 | Random walk length |
| | Walks/node | 10 | Number of walks per node |
| | p, q | 1.0, 1.0 | BFS/DFS exploration parameters |
| | Neg samples | 5 | Negative samples per positive |

## 📊 Training Results

### Current Implementation Results (Synthetic Data)

| Metric | Value | Paper Target |
|--------|-------|--------------|
| Best AUC | 72.01% | 84.32% |
| Final AUC | 70.80% | 84.32% |
| Final Accuracy | 66.33% | 77.68% |
| Final F1 Score | 63.27% | 78.11% |

*Note: Results are from synthetic data with a small knowledge graph (28 entities, 85 triples). Real StackOverflow data should yield results closer to paper targets.*

### Implementation Features

This implementation includes all key components from the paper with accurate algorithm implementations:

**Core Components:**
- ✅ **Double DQN** for EPDRL (proper action selection/evaluation separation)
- ✅ **Mean Pooling** for MDP state representation (Equation 5-6)
- ✅ **Gating Mechanism** for feature fusion (adaptive weighting)
- ✅ **Self-Attention** for question embeddings (richer representations)
- ✅ **Joint BCE + SSL Loss** training (Equation 19: L = L(θ) + β × L_con)

**Algorithm Accuracy:**
- ✅ **Node2Vec** with proper unigram^0.75 negative sampling distribution (Mikolov et al., 2013)
- ✅ **Hierarchical SSL** with two-level mutual information (global + subgraph)
- ✅ **TransH** hyperplane projection (Eq. 13-14) with margin-based ranking loss
- ✅ **Shaped reward function** with step penalties for path efficiency
- ✅ **Paper-aligned hyperparameters** documented from Section 5.2/Table 3

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
