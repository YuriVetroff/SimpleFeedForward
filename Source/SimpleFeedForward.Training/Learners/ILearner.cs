namespace SimpleFeedForward.Training.Learners
{
    public interface ILearner : IInitializing<INetwork>
    {
        void UpdateWeights();
    }
}
