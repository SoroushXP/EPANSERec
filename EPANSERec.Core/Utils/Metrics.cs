namespace EPANSERec.Core.Utils;

/// <summary>
/// Evaluation metrics for expert recommendation.
/// Implements metrics from Section 5.1.3 of the paper.
/// </summary>
public static class Metrics
{
    /// <summary>
    /// Computes Area Under ROC Curve (AUC).
    /// </summary>
    public static float ComputeAUC(List<float> predictions, List<int> labels)
    {
        var pairs = predictions.Zip(labels, (p, l) => (pred: p, label: l))
            .OrderByDescending(x => x.pred).ToList();
        
        int positives = labels.Count(l => l == 1);
        int negatives = labels.Count - positives;
        
        if (positives == 0 || negatives == 0) return 0.5f;
        
        float auc = 0;
        int tpCount = 0;
        
        foreach (var (pred, label) in pairs)
        {
            if (label == 1) tpCount++;
            else auc += tpCount;
        }
        
        return auc / (positives * negatives);
    }

    /// <summary>
    /// Computes Accuracy.
    /// </summary>
    public static float ComputeAccuracy(List<float> predictions, List<int> labels, float threshold = 0.5f)
    {
        int correct = 0;
        for (int i = 0; i < predictions.Count; i++)
        {
            int predicted = predictions[i] >= threshold ? 1 : 0;
            if (predicted == labels[i]) correct++;
        }
        return (float)correct / predictions.Count;
    }

    /// <summary>
    /// Computes Precision.
    /// </summary>
    public static float ComputePrecision(List<float> predictions, List<int> labels, float threshold = 0.5f)
    {
        int tp = 0, fp = 0;
        for (int i = 0; i < predictions.Count; i++)
        {
            int predicted = predictions[i] >= threshold ? 1 : 0;
            if (predicted == 1 && labels[i] == 1) tp++;
            if (predicted == 1 && labels[i] == 0) fp++;
        }
        return tp + fp > 0 ? (float)tp / (tp + fp) : 0;
    }

    /// <summary>
    /// Computes Recall.
    /// </summary>
    public static float ComputeRecall(List<float> predictions, List<int> labels, float threshold = 0.5f)
    {
        int tp = 0, fn = 0;
        for (int i = 0; i < predictions.Count; i++)
        {
            int predicted = predictions[i] >= threshold ? 1 : 0;
            if (predicted == 1 && labels[i] == 1) tp++;
            if (predicted == 0 && labels[i] == 1) fn++;
        }
        return tp + fn > 0 ? (float)tp / (tp + fn) : 0;
    }

    /// <summary>
    /// Computes F1 Score.
    /// </summary>
    public static float ComputeF1(List<float> predictions, List<int> labels, float threshold = 0.5f)
    {
        float precision = ComputePrecision(predictions, labels, threshold);
        float recall = ComputeRecall(predictions, labels, threshold);
        return precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
    }

    /// <summary>
    /// Computes Precision@K - precision of top K recommendations.
    /// </summary>
    public static float ComputePrecisionAtK(List<(float score, int label)> rankedResults, int k)
    {
        var topK = rankedResults.OrderByDescending(x => x.score).Take(k).ToList();
        int relevant = topK.Count(x => x.label == 1);
        return (float)relevant / k;
    }

    /// <summary>
    /// Computes Recall@K - recall of top K recommendations.
    /// </summary>
    public static float ComputeRecallAtK(List<(float score, int label)> rankedResults, int k)
    {
        var topK = rankedResults.OrderByDescending(x => x.score).Take(k).ToList();
        int relevantInTopK = topK.Count(x => x.label == 1);
        int totalRelevant = rankedResults.Count(x => x.label == 1);
        return totalRelevant > 0 ? (float)relevantInTopK / totalRelevant : 0;
    }

    /// <summary>
    /// Computes Mean Reciprocal Rank (MRR).
    /// </summary>
    public static float ComputeMRR(List<(float score, int label)> rankedResults)
    {
        var sorted = rankedResults.OrderByDescending(x => x.score).ToList();
        for (int i = 0; i < sorted.Count; i++)
        {
            if (sorted[i].label == 1)
                return 1.0f / (i + 1);
        }
        return 0;
    }

    /// <summary>
    /// Computes all metrics and returns as dictionary.
    /// </summary>
    public static Dictionary<string, float> ComputeAllMetrics(
        List<float> predictions, List<int> labels, int[] kValues = null!)
    {
        kValues ??= new[] { 1, 3, 5, 10 };
        
        var metrics = new Dictionary<string, float>
        {
            ["AUC"] = ComputeAUC(predictions, labels),
            ["Accuracy"] = ComputeAccuracy(predictions, labels),
            ["Precision"] = ComputePrecision(predictions, labels),
            ["Recall"] = ComputeRecall(predictions, labels),
            ["F1"] = ComputeF1(predictions, labels)
        };
        
        var rankedResults = predictions.Zip(labels, (p, l) => (score: p, label: l)).ToList();
        metrics["MRR"] = ComputeMRR(rankedResults);
        
        foreach (int k in kValues)
        {
            metrics[$"Precision@{k}"] = ComputePrecisionAtK(rankedResults, k);
            metrics[$"Recall@{k}"] = ComputeRecallAtK(rankedResults, k);
        }
        
        return metrics;
    }
}

