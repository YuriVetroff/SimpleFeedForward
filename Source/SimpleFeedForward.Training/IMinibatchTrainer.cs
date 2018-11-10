using SimpleFeedForward.Data;
using System.Collections.Generic;

namespace SimpleFeedForward.Training
{
    public interface IMinibatchTrainer
    {
        double TrainMinibatch(IEnumerable<DataItem> minibatch);
    }
}
