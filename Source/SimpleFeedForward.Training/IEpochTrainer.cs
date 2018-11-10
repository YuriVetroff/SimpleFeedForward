//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

using SimpleFeedForward.Data;
using System;
using System.Collections.Generic;

namespace SimpleFeedForward.Training
{
    public interface IEpochTrainer : IMinibatchTrainer
    {
        void Train(IEnumerable<DataItem> items);
        void Stop();
        event EventHandler<TrainingProgress> EpochPassed;
        event EventHandler<TrainingProgress> TrainingFinished;
    }
}
