using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace EPANSERec.Core.Recommendation;

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
/// </summary>
public class FeatureFusion : Module<Tensor, Tensor, Tensor>
{
    private readonly Linear _fusionLayer;
    private readonly int _embeddingDim;

    public FeatureFusion(int embeddingDim) : base("FeatureFusion")
    {
        _embeddingDim = embeddingDim;
        // Learnable fusion weights
        _fusionLayer = Linear(embeddingDim * 2, embeddingDim);
        RegisterComponents();
    }

    /// <summary>
    /// Fuses expertise preference features with semantic knowledge embeddings.
    /// </summary>
    /// <param name="expertisePreferenceEmb">Expertise preference embedding from GCN</param>
    /// <param name="semanticEmb">Semantic embedding from TransH</param>
    /// <returns>Fused embedding</returns>
    public override Tensor forward(Tensor expertisePreferenceEmb, Tensor semanticEmb)
    {
        // Concatenate and learn fusion weights
        var concatenated = torch.cat(new[] { expertisePreferenceEmb, semanticEmb }, dim: -1);
        var fused = functional.relu(_fusionLayer.forward(concatenated));
        return fused;
    }
}

