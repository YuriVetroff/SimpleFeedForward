namespace SimpleFeedForward.Training.StopConditions
{
    public interface IStopCondition
    {
        bool Stop(TrainingProgress progress);
    }
}
