namespace SimpleFeedForward.Training.Learners
{
    public interface ILearner
    {
        void UpdateWeights();
        void Init(INetwork network);
    }
}
