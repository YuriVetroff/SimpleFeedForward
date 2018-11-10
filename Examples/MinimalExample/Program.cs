using SimpleFeedForward;
using SimpleFeedForward.Data;
using SimpleFeedForward.Layers;
using SimpleFeedForward.Training;
using SimpleFeedForward.Training.Learners;
using SimpleFeedForward.Training.StopConditions;
using System;
using System.Collections.Generic;
using System.Linq;

using static System.Console;

namespace MinimalExample
{
    internal static class Program
    {
        private const int INPUT_SIZE = 100;
        private const int OUTPUT_SIZE = 4;
        private const int DATA_ITEMS_PER_CLASS = 10;
        private const double NOISE_PROBABILITY = 0.15;

        private static void Main(string[] args)
        {
            RunExample();
        }
        private static void RunExample()
        {
            var network = FluentNetwork.Create(INPUT_SIZE)
                .FullyConn(120, ActivationType.Relu)
                .FullyConn(100, ActivationType.Tanh)
                .FullyConn(OUTPUT_SIZE, ActivationType.Tanh)
                .Init();

            var trainer = new Trainer(network, GetConfig());
            trainer.EpochPassed += (o, e) =>
            {
                WriteLine($"Epoch {e.Epoch} passed. Minibatch = {e.MinibatchIndex}. Loss = {e.Loss}.");
            };
            trainer.TrainingFinished += (o, e) =>
            {
                WriteLine($"Training finished. Minibatch = {e.MinibatchIndex}. Loss = {e.Loss}.");
            };
            trainer.Train(GetDataset());

            ReadLine();
            trainer.Stop();
            
            ReadLine();
        }

        private static IEnumerable<DataItem> GetDataset()
        {
            var items = new List<DataItem>();
            var random = new Random();
            var valuePartSize = INPUT_SIZE / OUTPUT_SIZE;
            for (int classIndex = 0; classIndex < OUTPUT_SIZE; classIndex++)
            {
                for (int i = 0; i < DATA_ITEMS_PER_CLASS; i++)
                {
                    var item = DataItem.Create(INPUT_SIZE, OUTPUT_SIZE);
                    var startIndex = classIndex * valuePartSize;
                    for (int j = 0; j < valuePartSize; j++)
                    {
                        item.Input[startIndex + j] = 1;
                    }
                    for (int j = 0; j < INPUT_SIZE; j++)
                    {
                        if (random.NextDouble() <= NOISE_PROBABILITY)
                        {
                            item.Input[j] = random.NextDouble();
                        }
                    }
                    item.Output[classIndex] = 1;
                    items.Add(item);
                }
            }
            items = items.OrderBy(x => random.Next()).ToList();
            return items;
        }
        private static TrainingConfig GetConfig()
            => new TrainingConfig
            {
                MinibatchSize = 6,
                Learner = new MomentumLearner
                {
                    Momentum = 0.95
                },
                StopConditions = new List<IStopCondition>
                {
                    new SmallestRequiredLossStopCondition(0.01)
                }
            };
    }
}
