using SimpleFeedForward.Data;
using SimpleFeedForward.Layers;
using SimpleFeedForward.Training;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SimpleFeedForward
{
    public static class FluentNetwork
    {
        public static ILayerSequence Create(int inputSize)
            => new Network(inputSize);

        public static INetwork Init(this ILayerSequence sequence)
        {
            var network = sequence as INetwork;
            network.Init();
            return network;
        }

        public static INetwork Train(this INetwork network, TrainingConfig config, IEnumerable<DataItem> dataset)
        {
            var trainer = new Trainer(network, config);
            trainer.Train(dataset);
            return network;
        }

        public static ILayerSequence Tanh(this ILayerSequence sequence)
        {
            sequence.Add(new TanhLayer());
            return sequence;
        }

        public static ILayerSequence FullyConn(this ILayerSequence sequence, int neuronCount,
            ActivationType activationType = ActivationType.Undefined)
        {
            sequence.Add(new FullyConnectedLayer(neuronCount));

            if (activationType != ActivationType.Undefined)
            {
                sequence.Add(GetActivationLayer(activationType));
            }

            return sequence;
        }

        private static ActivationLayer GetActivationLayer(ActivationType activationType)
        {
            // The predicate to check if a type is decorated
            // with a single ActivationAttribute
            // and the value of the attribute's property
            // equals the searching activation type.
            Func<Type, bool> hasTargetAttribute = type =>
            {
                // Here we get all activation attributes the type decorated with.
                var activationAttributes = type
                    .GetCustomAttributes(typeof(ActivationAttribute), true)
                    .Select(attr => attr as ActivationAttribute);

                // If there is not only one activation attribute,
                // we return false.
                if (activationAttributes.Count() != 1)
                {
                    return false;
                }

                // Gets the single activation attribute.
                var activationAttribute = activationAttributes.First();

                // If the ActivationType property of the attribute doesn't
                // equal the searching activation type, return false.
                if (activationAttribute.ActivationType != activationType)
                {
                    return false;
                }

                // Return true if we have successfully passed all the steps above.
                return true;
            };

            // Get types of all activation layers that are decorated
            // with the target attribute.
            var activationLayersTypes = AppDomain.CurrentDomain.GetAssemblies()
                .SelectMany(x => x.GetTypes())
                .Where(t => t.BaseType == typeof(ActivationLayer))
                .Where(t => t.IsDefined(typeof(ActivationAttribute), true))
                .Where(hasTargetAttribute);

            // Get the number of the layer types.
            // We have to find only one.
            var count = activationLayersTypes.Count();
            if (count > 1)
            {
                throw new Exception(
                    $"Found more than one layer with the {activationType} activation type!");
            }
            if (count < 1)
            {
                throw new NotImplementedException(
                    $"A layer for the {activationType} activation type has not been implemented yet!");
            }

            // Use the Activator class to make the method happy.
            return Activator.CreateInstance(activationLayersTypes.First()) as ActivationLayer;
        }
    }
}
