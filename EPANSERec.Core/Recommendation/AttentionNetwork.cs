using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace EPANSERec.Core.Recommendation;

/// <summary>
/// Self-Attention Network for computing attention-weighted question embeddings.
/// Uses self-attention over question's entity embeddings to create a richer representation.
/// </summary>
public class QuestionAttentionNetwork : Module<Tensor, Tensor>
{
    private readonly Linear _queryLayer;
    private readonly Linear _keyLayer;
    private readonly Linear _valueLayer;
    private readonly int _embeddingDim;
    private readonly float _scale;

    public QuestionAttentionNetwork(int embeddingDim) : base("QuestionAttentionNetwork")
    {
        _embeddingDim = embeddingDim;
        _scale = (float)Math.Sqrt(embeddingDim);

        // Self-attention projection layers
        _queryLayer = Linear(embeddingDim, embeddingDim);
        _keyLayer = Linear(embeddingDim, embeddingDim);
        _valueLayer = Linear(embeddingDim, embeddingDim);

        RegisterComponents();
    }

    /// <summary>
    /// Computes self-attention weighted question embedding.
    /// Uses scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d)V
    /// </summary>
    /// <param name="entityEmbeddings">Entity embeddings for the question [n, embeddingDim]</param>
    /// <returns>Attention-weighted question embedding [embeddingDim]</returns>
    public override Tensor forward(Tensor entityEmbeddings)
    {
        if (entityEmbeddings.size(0) == 0)
            return torch.zeros(_embeddingDim);

        if (entityEmbeddings.size(0) == 1)
            return entityEmbeddings.squeeze(0);

        // Compute Q, K, V projections
        var queries = _queryLayer.forward(entityEmbeddings);  // [n, d]
        var keys = _keyLayer.forward(entityEmbeddings);       // [n, d]
        var values = _valueLayer.forward(entityEmbeddings);   // [n, d]

        // Compute attention scores: QK^T / √d
        var attentionScores = torch.matmul(queries, keys.transpose(0, 1)) / _scale;  // [n, n]
        var attentionWeights = functional.softmax(attentionScores, dim: -1);  // [n, n]

        // Apply attention to values
        var attended = torch.matmul(attentionWeights, values);  // [n, d]

        // Mean pool over attended representations for final question embedding
        return attended.mean(dimensions: new long[] { 0 });  // [d]
    }
}

/// <summary>
/// Attention Network for weighting expert's historical Q&A relevance to question.
/// Implements Equation 15-16 from the paper.
/// </summary>
public class AttentionNetwork : Module<Tensor, Tensor, Tensor>
{
    private readonly int _embeddingDim;

    public AttentionNetwork(int embeddingDim) : base("AttentionNetwork")
    {
        _embeddingDim = embeddingDim;
        RegisterComponents();
    }

    /// <summary>
    /// Computes attention-weighted expert representation.
    /// </summary>
    /// <param name="questionEmbedding">Question embedding e(q') [embeddingDim]</param>
    /// <param name="historicalQAEmbeddings">Expert's historical Q&A embeddings [m, embeddingDim]</param>
    /// <returns>Weighted expert representation e(ui) [embeddingDim]</returns>
    public override Tensor forward(Tensor questionEmbedding, Tensor historicalQAEmbeddings)
    {
        // Ensure question embedding has correct shape for dot product
        var questionExp = questionEmbedding.unsqueeze(0); // [1, embeddingDim]
        
        // Compute attention scores: a_k = exp(e(q') · e(q_k)) / Σ exp(e(q') · e(q_j))
        // Equation 15
        var dotProducts = torch.matmul(historicalQAEmbeddings, questionExp.transpose(0, 1)); // [m, 1]
        var attentionWeights = functional.softmax(dotProducts, dim: 0); // [m, 1]
        
        // Compute weighted sum: e(ui) = Σ a_k · e(q_k)
        // Equation 16
        var expertRepresentation = (historicalQAEmbeddings * attentionWeights).sum(dim: 0); // [embeddingDim]
        
        return expertRepresentation;
    }

    /// <summary>
    /// Computes attention weights for interpretability.
    /// </summary>
    public float[] GetAttentionWeights(Tensor questionEmbedding, Tensor historicalQAEmbeddings)
    {
        using (torch.no_grad())
        {
            var questionExp = questionEmbedding.unsqueeze(0);
            var dotProducts = torch.matmul(historicalQAEmbeddings, questionExp.transpose(0, 1));
            var weights = functional.softmax(dotProducts, dim: 0);
            return weights.data<float>().ToArray();
        }
    }
}

/// <summary>
/// Feature Fusion module for combining expertise preference and semantic embeddings.
/// Section 4.3 of the paper.
/// Uses a gating mechanism to adaptively fuse features:
/// g = σ(W_g · [e_pref; e_sem] + b_g)  -- gate
/// e_fused = g ⊙ e_pref + (1-g) ⊙ e_sem  -- gated fusion
/// </summary>
public class FeatureFusion : Module<Tensor, Tensor, Tensor>
{
    private readonly Linear _gateLayer;        // Computes gating weights
    private readonly Linear _transformPref;    // Transform preference embedding
    private readonly Linear _transformSem;     // Transform semantic embedding
    private readonly int _embeddingDim;

    public FeatureFusion(int embeddingDim) : base("FeatureFusion")
    {
        _embeddingDim = embeddingDim;

        // Gate layer: takes concatenated embeddings, outputs gate values [0,1]
        _gateLayer = Linear(embeddingDim * 2, embeddingDim);

        // Transform layers to project both embeddings to same space
        _transformPref = Linear(embeddingDim, embeddingDim);
        _transformSem = Linear(embeddingDim, embeddingDim);

        RegisterComponents();
    }

    /// <summary>
    /// Fuses expertise preference features with semantic knowledge embeddings using gating.
    /// Gate mechanism allows the model to adaptively weight contributions from each source.
    /// </summary>
    /// <param name="expertisePreferenceEmb">Expertise preference embedding from GCN</param>
    /// <param name="semanticEmb">Semantic embedding from TransH</param>
    /// <returns>Fused embedding</returns>
    public override Tensor forward(Tensor expertisePreferenceEmb, Tensor semanticEmb)
    {
        // Transform both embeddings
        var prefTransformed = functional.tanh(_transformPref.forward(expertisePreferenceEmb));
        var semTransformed = functional.tanh(_transformSem.forward(semanticEmb));

        // Compute gate: g = σ(W_g · [e_pref; e_sem] + b_g)
        var concatenated = torch.cat(new[] { expertisePreferenceEmb, semanticEmb }, dim: -1);
        var gate = functional.sigmoid(_gateLayer.forward(concatenated));

        // Gated fusion: e_fused = g ⊙ e_pref + (1-g) ⊙ e_sem
        var fused = gate * prefTransformed + (1 - gate) * semTransformed;

        return fused;
    }
}

