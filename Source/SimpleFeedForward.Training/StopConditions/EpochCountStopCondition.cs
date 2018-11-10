namespace SimpleFeedForward.Training.StopConditions
{
    public class EpochCountStopCondition : IStopCondition
    {
        public EpochCountStopCondition(int maxEpochs)
        {
            MaxEpochs = maxEpochs;
        }

        public int MaxEpochs { get; }

        public bool Stop(TrainingProgress progress)
            => progress.Epoch >= MaxEpochs ? true : false;
    }
}
