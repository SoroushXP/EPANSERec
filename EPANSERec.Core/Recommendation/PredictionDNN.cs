using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace EPANSERec.Core.Recommendation;

/// <summary>
/// Deep Neural Network for predicting expert-question match probability.
/// Implements Equation 17-18 from the paper.
/// </summary>
public class PredictionDNN : Module<Tensor, Tensor>
{
    private readonly Linear _fc1;
    private readonly Linear _fc2;
    private readonly Linear _fc3;
    private readonly Linear _output;
    private readonly Dropout _dropout;
    private readonly BatchNorm1d _bn1;
    private readonly BatchNorm1d _bn2;

    public PredictionDNN(int inputDim, int hiddenDim = 256, float dropoutRate = 0.2f) 
        : base("PredictionDNN")
    {
        // Input: concatenation of question and expert embeddings [e(q'); e(ui)]
        _fc1 = Linear(inputDim, hiddenDim);
        _bn1 = BatchNorm1d(hiddenDim);
        _fc2 = Linear(hiddenDim, hiddenDim / 2);
        _bn2 = BatchNorm1d(hiddenDim / 2);
        _fc3 = Linear(hiddenDim / 2, hiddenDim / 4);
        _output = Linear(hiddenDim / 4, 1);
        _dropout = Dropout(dropoutRate);
        
        RegisterComponents();
    }

    /// <summary>
    /// Forward pass to predict match probability.
    /// ŷ_i = ReLU(W_l * [e(q'); e(ui)] + b_l)
    /// Final output through Sigmoid for [0,1] probability.
    /// </summary>
    public override Tensor forward(Tensor input)
    {
        var x = _fc1.forward(input);

        // BatchNorm requires batch size > 1 during training
        // Skip batch norm for single samples
        if (input.shape[0] > 1 && training)
        {
            x = _bn1.forward(x);
        }
        x = functional.relu(x);
        x = _dropout.forward(x);

        x = _fc2.forward(x);
        if (input.shape[0] > 1 && training)
        {
            x = _bn2.forward(x);
        }
        x = functional.relu(x);
        x = _dropout.forward(x);

        x = functional.relu(_fc3.forward(x));
        x = functional.sigmoid(_output.forward(x)); // Sigmoid for probability
        return x;
    }
}

/// <summary>
/// Binary Cross Entropy loss for expert recommendation (Equation 18).
/// L(θ) = -Σ[y_i * log(ŷ_i) + (1-y_i) * log(1-ŷ_i)]
/// </summary>
public static class RecommendationLoss
{
    /// <summary>
    /// Computes recommendation loss combining BCE and SSL loss.
    /// L = L(θ) + β * L_con (Equation 19)
    /// </summary>
    public static Tensor ComputeTotalLoss(Tensor predictions, Tensor labels, Tensor sslLoss, float beta = 0.1f)
    {
        var bceLoss = functional.binary_cross_entropy(predictions, labels);
        return bceLoss + beta * sslLoss;
    }

    /// <summary>
    /// Computes binary cross entropy loss only.
    /// </summary>
    public static Tensor ComputeBCELoss(Tensor predictions, Tensor labels)
    {
        return functional.binary_cross_entropy(predictions, labels);
    }
}

