using EPANSERec.Core.KnowledgeGraph;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace EPANSERec.Core.Embeddings;

/// <summary>
/// TransH model for knowledge graph embedding with semantic information.
/// Implements Section 4.3 (Equations 13-14) from the EPAN-SERec paper.
///
/// TransH (Wang et al., 2014) projects entities onto relation-specific hyperplanes,
/// allowing different representations of entities for different relations.
///
/// ═══════════════════════════════════════════════════════════════════════════════
/// MATHEMATICAL FORMULATION
/// ═══════════════════════════════════════════════════════════════════════════════
///
/// **Projection onto hyperplane (Equation 13):**
///   h⊥ = h - w_r^T · h · w_r
///   t⊥ = t - w_r^T · t · w_r
///
/// where w_r is the normal vector of relation r's hyperplane (||w_r|| = 1)
///
/// **Score function:**
///   f_r(h, t) = ||h⊥ + d_r - t⊥||²
///
/// where d_r is the translation vector for relation r
///
/// **Margin-based ranking loss (Equation 14):**
///   L = Σ max(0, f_r(h, t) - f_r(h', t') + γ)
///
/// where (h', t') is a corrupted triple (negative sample) and γ is the margin
///
/// ═══════════════════════════════════════════════════════════════════════════════
/// HYPERPARAMETERS (Paper Section 5.2)
/// ═══════════════════════════════════════════════════════════════════════════════
/// - Embedding dimension: 100 (default, range 50-200)
/// - Margin γ: 1.0 (standard for TransH)
/// - Learning rate: 0.01 (SGD or Adam)
/// - Negative samples per positive: 1 (paper) or more for better training
/// ═══════════════════════════════════════════════════════════════════════════════
/// </summary>
public class TransH : Module
{
    private readonly Embedding _entityEmbeddings;
    private readonly Embedding _relationEmbeddings;  // Translation vectors (d_r)
    private readonly Embedding _normalVectors;       // Hyperplane normals (w_r), ||w_r|| = 1
    private readonly int _embeddingDim;
    private readonly int _numEntities;
    private readonly int _numRelations;
    private readonly float _margin;                  // γ in margin-based ranking loss
    private torch.optim.Optimizer? _optimizer;

    /// <summary>
    /// Creates a TransH model for knowledge graph embedding.
    /// </summary>
    /// <param name="numEntities">Total number of entities in KG</param>
    /// <param name="numRelations">Total number of relation types</param>
    /// <param name="embeddingDim">Embedding dimension (paper default: 100)</param>
    /// <param name="margin">Margin γ for ranking loss (paper default: 1.0)</param>
    public TransH(int numEntities, int numRelations, int embeddingDim = 100, float margin = 1.0f)
        : base("TransH")
    {
        _numEntities = numEntities;
        _numRelations = numRelations;
        _embeddingDim = embeddingDim;
        _margin = margin;

        // Entity embeddings: h, t ∈ R^d
        _entityEmbeddings = Embedding(numEntities, embeddingDim);

        // Relation translation vectors: d_r ∈ R^d
        _relationEmbeddings = Embedding(numRelations, embeddingDim);

        // Hyperplane normal vectors: w_r ∈ R^d with ||w_r|| = 1
        _normalVectors = Embedding(numRelations, embeddingDim);

        RegisterComponents();

        // Initialize embeddings with Xavier uniform
        InitializeEmbeddings();
    }

    private void InitializeEmbeddings()
    {
        init.xavier_uniform_(_entityEmbeddings.weight);
        init.xavier_uniform_(_relationEmbeddings.weight);
        init.xavier_uniform_(_normalVectors.weight);

        // Normalize normal vectors to unit norm
        using (torch.no_grad())
        {
            var normalizedNormals = NormalizeL2(_normalVectors.weight, dim: 1);
            _normalVectors.weight.copy_(normalizedNormals);
        }
    }

    /// <summary>
    /// L2 normalization helper function.
    /// </summary>
    private static Tensor NormalizeL2(Tensor input, int dim = -1)
    {
        // Compute L2 norm manually
        var squared = input * input;
        var summed = squared.sum(new long[] { dim }, keepdim: true);
        var norm = torch.sqrt(summed);
        return input / (norm + 1e-10f);
    }

    /// <summary>
    /// Projects entity onto relation-specific hyperplane.
    /// h⊥ = h - w_r^T * h * w_r
    /// </summary>
    private Tensor ProjectToHyperplane(Tensor entityEmb, Tensor normalVector)
    {
        // Ensure normal vector is normalized
        var normalizedNormal = NormalizeL2(normalVector, dim: -1);

        // Compute projection: h - (h · w_r) * w_r
        var dotProduct = (entityEmb * normalizedNormal).sum(dim: -1, keepdim: true);
        var projection = entityEmb - dotProduct * normalizedNormal;

        return projection;
    }

    /// <summary>
    /// Computes TransH score function (Equation 13).
    /// f_r(h, t) = ||h⊥ + d_r - t⊥||^2
    /// </summary>
    public Tensor ScoreFunction(Tensor headIds, Tensor relationIds, Tensor tailIds)
    {
        var headEmb = _entityEmbeddings.forward(headIds);
        var tailEmb = _entityEmbeddings.forward(tailIds);
        var relationEmb = _relationEmbeddings.forward(relationIds);
        var normalVec = _normalVectors.forward(relationIds);
        
        // Project head and tail to hyperplane
        var headProjected = ProjectToHyperplane(headEmb, normalVec);
        var tailProjected = ProjectToHyperplane(tailEmb, normalVec);
        
        // Score: ||h⊥ + d_r - t⊥||^2
        var score = torch.pow(headProjected + relationEmb - tailProjected, 2).sum(dim: -1);
        
        return score;
    }

    /// <summary>
    /// Computes margin-based ranking loss (Equation 14).
    /// L = Σ max(0, f_r(h,t) + γ - f_r(h',t'))
    /// </summary>
    public Tensor ComputeLoss(Tensor posHeads, Tensor posRelations, Tensor posTails,
                              Tensor negHeads, Tensor negRelations, Tensor negTails)
    {
        var posScores = ScoreFunction(posHeads, posRelations, posTails);
        var negScores = ScoreFunction(negHeads, negRelations, negTails);
        
        // Margin ranking loss
        var loss = functional.relu(posScores + _margin - negScores).mean();
        
        return loss;
    }

    /// <summary>
    /// Trains the TransH model on knowledge graph triples.
    /// </summary>
    public void Train(SoftwareKnowledgeGraph kg, int epochs = 100, int batchSize = 128,
        float learningRate = 0.01f)
    {
        _optimizer = torch.optim.Adam(parameters(), lr: learningRate);
        var random = new Random();
        var entityIds = kg.Entities.Keys.ToList();
        var startTime = DateTime.Now;

        Console.WriteLine($"  Training on {kg.Triples.Count} triples...");
        Console.Out.Flush();

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            float epochLoss = 0;
            var triples = kg.Triples.OrderBy(_ => random.Next()).ToList();

            for (int i = 0; i < triples.Count; i += batchSize)
            {
                var batch = triples.Skip(i).Take(batchSize).ToList();
                epochLoss += TrainBatch(batch, entityIds, random);
            }

            // Progress reporting - more frequent at start, then every 20 epochs
            bool shouldReport = epoch == 0 || (epoch + 1) % 20 == 0 || epoch == epochs - 1;
            if (shouldReport)
            {
                var elapsed = DateTime.Now - startTime;
                var avgTimePerEpoch = elapsed.TotalSeconds / (epoch + 1);
                var remaining = TimeSpan.FromSeconds(avgTimePerEpoch * (epochs - epoch - 1));
                int numBatches = (triples.Count + batchSize - 1) / batchSize;
                float avgLoss = numBatches > 0 ? epochLoss / numBatches : 0;
                Console.WriteLine($"  TransH Epoch {epoch + 1}/{epochs} - Loss: {avgLoss:F4} - ETA: {remaining:mm\\:ss}");
                Console.Out.Flush();
            }
        }
    }

    private float TrainBatch(List<Triple> batch, List<int> entityIds, Random random)
    {
        _optimizer!.zero_grad();

        int n = batch.Count;
        var posHeads = new long[n];
        var posRelations = new long[n];
        var posTails = new long[n];
        var negHeads = new long[n];
        var negRelations = new long[n];
        var negTails = new long[n];

        for (int i = 0; i < n; i++)
        {
            var triple = batch[i];
            posHeads[i] = triple.Head.Id;
            posRelations[i] = triple.Relation.Id;
            posTails[i] = triple.Tail.Id;

            // Create negative sample by corrupting head or tail
            if (random.NextDouble() < 0.5)
            {
                negHeads[i] = entityIds[random.Next(entityIds.Count)];
                negRelations[i] = triple.Relation.Id;
                negTails[i] = triple.Tail.Id;
            }
            else
            {
                negHeads[i] = triple.Head.Id;
                negRelations[i] = triple.Relation.Id;
                negTails[i] = entityIds[random.Next(entityIds.Count)];
            }
        }

        using var posHeadsTensor = torch.tensor(posHeads);
        using var posRelsTensor = torch.tensor(posRelations);
        using var posTailsTensor = torch.tensor(posTails);
        using var negHeadsTensor = torch.tensor(negHeads);
        using var negRelsTensor = torch.tensor(negRelations);
        using var negTailsTensor = torch.tensor(negTails);

        var loss = ComputeLoss(posHeadsTensor, posRelsTensor, posTailsTensor,
                               negHeadsTensor, negRelsTensor, negTailsTensor);

        loss.backward();
        _optimizer.step();

        // Normalize normal vectors after each update
        using (torch.no_grad())
        {
            var normalizedNormals = NormalizeL2(_normalVectors.weight, dim: 1);
            _normalVectors.weight.copy_(normalizedNormals);
        }

        return loss.item<float>();
    }

    /// <summary>
    /// Gets the embedding for an entity.
    /// </summary>
    public float[] GetEntityEmbedding(int entityId)
    {
        using (torch.no_grad())
        {
            using var idTensor = torch.tensor(new long[] { entityId });
            var embedding = _entityEmbeddings.forward(idTensor);
            return embedding.data<float>().ToArray();
        }
    }

    /// <summary>
    /// Gets embeddings for all entities.
    /// </summary>
    public Dictionary<int, float[]> GetAllEntityEmbeddings(IEnumerable<int> entityIds)
    {
        var result = new Dictionary<int, float[]>();
        using (torch.no_grad())
        {
            foreach (var entityId in entityIds)
            {
                result[entityId] = GetEntityEmbedding(entityId);
            }
        }
        return result;
    }

    /// <summary>
    /// Gets the relation embedding (translation vector).
    /// </summary>
    public float[] GetRelationEmbedding(int relationId)
    {
        using (torch.no_grad())
        {
            using var idTensor = torch.tensor(new long[] { relationId });
            var embedding = _relationEmbeddings.forward(idTensor);
            return embedding.data<float>().ToArray();
        }
    }
}
