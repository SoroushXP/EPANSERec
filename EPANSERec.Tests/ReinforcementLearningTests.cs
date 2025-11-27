using EPANSERec.Core.ReinforcementLearning;

namespace EPANSERec.Tests;

/// <summary>
/// Unit tests for Reinforcement Learning components.
/// </summary>
public class ReinforcementLearningTests
{
    [Fact]
    public void MDPState_Creation_ShouldInitializeEmpty()
    {
        // Arrange & Act
        var state = new MDPState(embeddingDimension: 64);
        
        // Assert
        Assert.Equal(64, state.EmbeddingDimension);
        Assert.Empty(state.PathSequence);
        Assert.Empty(state.StateVector);
    }

    [Fact]
    public void MDPState_AddNode_ShouldUpdatePathAndVector()
    {
        // Arrange
        var state = new MDPState(embeddingDimension: 4);
        var embedding = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        
        // Act
        state.AddNode(entityId: 5, nodeEmbedding: embedding);
        
        // Assert
        Assert.Single(state.PathSequence);
        Assert.Equal(5, state.PathSequence[0]);
        Assert.Equal(4, state.StateVector.Length);
    }

    [Fact]
    public void MDPState_AddMultipleNodes_ShouldConcatenateVectors()
    {
        // Arrange
        var state = new MDPState(embeddingDimension: 3);
        var emb1 = new float[] { 1.0f, 2.0f, 3.0f };
        var emb2 = new float[] { 4.0f, 5.0f, 6.0f };
        
        // Act
        state.AddNode(1, emb1);
        state.AddNode(2, emb2);
        
        // Assert
        Assert.Equal(2, state.PathSequence.Count);
        Assert.Equal(6, state.StateVector.Length); // 3 + 3
    }

    [Fact]
    public void MDPState_CurrentNode_ShouldReturnLastNode()
    {
        // Arrange
        var state = new MDPState(embeddingDimension: 4);
        state.AddNode(1, new float[4]);
        state.AddNode(2, new float[4]);
        state.AddNode(3, new float[4]);
        
        // Act & Assert
        Assert.Equal(3, state.CurrentNode);
    }

    [Fact]
    public void MDPState_ContainsEntity_ShouldReturnTrueIfInPath()
    {
        // Arrange
        var state = new MDPState(embeddingDimension: 4);
        state.AddNode(1, new float[4]);
        state.AddNode(2, new float[4]);
        
        // Act & Assert
        Assert.True(state.ContainsEntity(1));
        Assert.True(state.ContainsEntity(2));
        Assert.False(state.ContainsEntity(3));
    }

    [Fact]
    public void MDPState_GetPooledStateVector_ShouldReturnCorrectSize()
    {
        // Arrange
        int dim = 4;
        var state = new MDPState(embeddingDimension: dim);
        state.AddNode(1, new float[] { 1, 2, 3, 4 });
        state.AddNode(2, new float[] { 5, 6, 7, 8 });
        
        // Act
        var pooled = state.GetPooledStateVector();
        
        // Assert
        Assert.Equal(dim * 2, pooled.Length); // pooled + current
    }

    [Fact]
    public void MDPState_Clone_ShouldCreateIndependentCopy()
    {
        // Arrange
        var state = new MDPState(embeddingDimension: 4);
        state.AddNode(1, new float[] { 1, 2, 3, 4 });
        
        // Act
        var clone = state.Clone();
        state.AddNode(2, new float[] { 5, 6, 7, 8 });
        
        // Assert
        Assert.Single(clone.PathSequence);
        Assert.Equal(2, state.PathSequence.Count);
    }

    [Fact]
    public void Experience_Creation_ShouldSetAllProperties()
    {
        // Arrange
        var stateVec = new float[] { 1, 2, 3 };
        var nextStateVec = new float[] { 4, 5, 6 };
        
        // Act
        var exp = new Experience(stateVec, action: 2, reward: 1.0f, nextStateVec, done: true);
        
        // Assert
        Assert.Equal(stateVec, exp.State);
        Assert.Equal(2, exp.Action);
        Assert.Equal(1.0f, exp.Reward);
        Assert.Equal(nextStateVec, exp.NextState);
        Assert.True(exp.Done);
    }

    [Fact]
    public void ReplayMemory_Push_ShouldAddExperience()
    {
        // Arrange
        var memory = new ReplayMemory(capacity: 100);
        var exp = new Experience(new float[3], 0, 1.0f, new float[3], false);
        
        // Act
        memory.Push(exp);
        
        // Assert
        Assert.Equal(1, memory.Count);
    }

    [Fact]
    public void ReplayMemory_Sample_ShouldReturnRequestedSize()
    {
        // Arrange
        var memory = new ReplayMemory(capacity: 100, seed: 42);
        for (int i = 0; i < 50; i++)
        {
            memory.Push(new Experience(new float[3], i % 4, 0.5f, new float[3], false));
        }
        
        // Act
        var batch = memory.Sample(batchSize: 10);
        
        // Assert
        Assert.Equal(10, batch.Count);
    }

    [Fact]
    public void ReplayMemory_CanSample_ShouldReturnFalseIfNotEnough()
    {
        // Arrange
        var memory = new ReplayMemory(capacity: 100);
        memory.Push(new Experience(new float[3], 0, 1.0f, new float[3], false));
        
        // Act & Assert
        Assert.False(memory.CanSample(10));
        Assert.True(memory.CanSample(1));
    }
}

