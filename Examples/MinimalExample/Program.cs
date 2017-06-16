//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

using SimpleFeedForward;
using SimpleFeedForward.Layers;
using SimpleFeedForward.Training;
using System;
using System.Collections.Generic;
using System.Linq;
using static System.Console;

namespace MinimalExample
{
    internal static class Program
    {
        #region Constants

        /// <summary>
        ///     The number of input neurons.
        /// </summary>
        private const int InputCount = 100;

        /// <summary>
        ///     The number of output neurons (or the number of classes).
        /// </summary>
        private const int OutputCount = 4;

        /// <summary>
        ///     The number of class we are going to train.
        /// </summary>
        private const int ClassToBeTrained = 2;

        /// <summary>
        ///     The number of epochs we are going to run.
        /// </summary>
        private const int EpochCount = 250;

        #endregion

        private static void Main(string[] args)
        {
            // It's a very simple example how SimpleFeedForward engine works.

            // The code below creates a network of fully-connected and activation layers.
            // We run the network and look for its random-based input.

            // Then we initialize a trainer and run the network firstly after one,
            // and then after several (100+) training epochs.

            // The results changes towards the number of the class we are training.

            WriteLine("SimpleFeedForward: Minimal example");
            
            WriteLine($"Input count = {InputCount}");
            WriteLine($"Class count = {OutputCount}");
            WriteLine($"Training class = {ClassToBeTrained}");

            // Create the network.
            var network = FastNetwork.Create(InputCount)
                .FullyConn(120, ActivationType.Relu)
                .FullyConn(100, ActivationType.Tanh)
                .FullyConn(OutputCount, ActivationType.Tanh)
                .Init();

            // Create the input using Random.
            var random = new Random();
            var input = Enumerable.Range(0, InputCount)
                .Select(i => random.NextDouble())
                .ToArray();

            // Create the expected output for the specified class
            // (the ClassToBeTrained constant).
            var expectedOutput = new double[OutputCount];
            expectedOutput[ClassToBeTrained] = 1;
            
            // Run the network before any training.
            // We get an output array based on initial random values.
            network.Run(input, "The output before any training:");

            // Now let's train the network.
            // Here we create and initialize the momentum SGD trainer.
            var trainer = new MomentumTrainer();
            trainer.Init(network);

            // Perform 1 training iteration.
            // After the training the response for the class #0 is a little bit higher.
            trainer.Train(input, expectedOutput);
            network.Run(input, "The output after 1 training epoch:");

            // Now let's run several training iterations and look to the output.
            // The response for the class we trained has become rather higher than before.
            var epochIndex = 0;
            do trainer.Train(input, expectedOutput);
            while (epochIndex++ < EpochCount);
            network.Run(input, $"The output after {EpochCount} training epochs:");

            WriteLine();
            WriteLine("Press any key to exit.");
            ReadLine();
        }

        private static void Print(this IEnumerable<double> array, string message = "")
        {
            WriteLine();

            if (!string.IsNullOrWhiteSpace(message))
                WriteLine(message);

            foreach (var item in array)
                WriteLine("{0:0.0000}", item);
        }
        private static void Run(this INetwork network, double[] input, string message = "")
        {
            network.Forward(input);
            network.Output.AsEnumerable().Print(message);
        }
    }
}
