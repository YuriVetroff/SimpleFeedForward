namespace SimpleFeedForward.Training.StopConditions
{
    public class SmallestRequiredLossStopCondition : IStopCondition
    {
        public SmallestRequiredLossStopCondition(double smallestRequiredLoss)
        {
            SmallestRequiredLoss = smallestRequiredLoss;
        }

        public double SmallestRequiredLoss { get; }

        public bool Stop(TrainingProgress progress)
            => progress.Loss <= SmallestRequiredLoss ? true : false;
    }
}
