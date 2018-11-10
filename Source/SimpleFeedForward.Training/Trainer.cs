using SimpleFeedForward.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SimpleFeedForward.Training
{
    public class Trainer : IEpochTrainer
    {
        protected readonly INetwork _network;
        protected readonly TrainingConfig _config;
        private bool _isStopped = false;

        public Trainer(INetwork network, TrainingConfig config)
        {
            _network = network;
            _config = config;
            _config.Learner.Init(_network);
        }

        public async void Train(IEnumerable<DataItem> items)
        {
            await Task.Run(() =>
            {
                _isStopped = false;

                var minibatchSource = new MinibatchSource(items);

                var progress = new TrainingProgress();

                while (true)
                {
                    if (_isStopped)
                    {
                        break;
                    }

                    var batch = minibatchSource.NextMinibatch(_config.MinibatchSize);
                    progress.Loss = TrainMinibatch(batch);

                    progress.MinibatchIndex++;
                    if (minibatchSource.IsSweepEnd)
                    {
                        OnEpochPassed(progress);
                        progress.Epoch++;
                    }

                    if (_config.StopConditions.Any(x => x.Stop(progress)))
                    {
                        break;
                    }
                }
                OnTrainingFinished(progress);
            });
        }
        public double TrainMinibatch(IEnumerable<DataItem> items)
        {
            var loss = 0d;
            foreach (var item in items)
            {
                _network.Forward(item.Input);
                var expected = item.Output;
                var actual = _network.Output.ToArray();
                var gradient = ComputeGradient(expected, actual);
                loss += ComputeLoss(expected, actual);

                _network.Backward(gradient);
            }
            _config.Learner.UpdateWeights();

            loss /= _config.MinibatchSize;
            return loss;
        }

        public void Stop() => _isStopped = true;

        private double[] ComputeGradient(double[] expected, double[] actual)
        {
            var outputLength = expected.Length;
            var gradient = new double[outputLength];
            for (int i = 0; i < outputLength; i++)
            {
                gradient[i] = actual[i] - expected[i];
            }
            return gradient;
        }
        private double ComputeLoss(double[] expected, double[] actual)
        {
            var loss = 0d;
            for (int i = 0; i < expected.Length; i++)
            {
                loss += Math.Pow(expected[i] - actual[i], 2);
            }
            loss /= 2;
            return loss;
        }
        
        public event EventHandler<TrainingProgress> EpochPassed;
        public event EventHandler<TrainingProgress> TrainingFinished;

        protected void OnEpochPassed(TrainingProgress progress)
            => EpochPassed?.Invoke(this, progress);
        protected void OnTrainingFinished(TrainingProgress progress)
            => TrainingFinished?.Invoke(this, progress);

    }
}
