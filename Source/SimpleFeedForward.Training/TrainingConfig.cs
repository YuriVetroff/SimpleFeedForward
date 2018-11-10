using SimpleFeedForward.Training.Learners;
using SimpleFeedForward.Training.StopConditions;
using System.Collections.Generic;

namespace SimpleFeedForward.Training
{
    public class TrainingConfig
    {
        public int MinibatchSize { get; set; }
        public ILearner Learner { get; set; }
        public IEnumerable<IStopCondition> StopConditions { get; set; }
    }
}
