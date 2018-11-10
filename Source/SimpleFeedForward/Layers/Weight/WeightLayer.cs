using System;
using System.Collections.Generic;

namespace SimpleFeedForward.Layers
{
    public interface IWeightLayer : ILayer, ITrainable
    {
        int WeightsLength { get; }
        int WeightsPerNeuron { get; }

        IReadOnlyList<double> Weights { get; }

        void SetWeights(double[] weights);
    }

    public abstract class WeightLayer : Layer, IWeightLayer
    {
        protected double[] _weights;
        protected double[] _weightsGradient;

        protected int _weightsLength;
        protected int _weightsPerNeuron;

        public int WeightsLength => _weightsLength;
        public int WeightsPerNeuron => _weightsLength / NeuronCount;

        public IReadOnlyList<double> Weights => _weights;

        public override void Forward()
        {
            base.Forward();

            _engine.ForwardDotProduct(_input, _weights, _output);
        }
        public override void Backward()
        {
            _engine.BackwardDotProduct(_input, _backError, _weights, _weightsGradient, _error);

            base.Backward();
        }

        public override void Init(int incomingLength)
        {
            base.Init(incomingLength);

            _weightsLength = _incomingLength * _outgoingLength;
            _weightsPerNeuron = _weightsLength / NeuronCount;

            _weights = new double[_weightsLength];
            _weightsGradient = new double[_weightsLength];

            var random = new Random();
            for (int i = 0; i < _weightsLength; i++)
            {
                _weights[i] = random.NextDouble() / _incomingLength;
            }
        }

        public virtual IEnumerable<DataFlow> ExtractData()
        {
            var data = new List<DataFlow>();

            data.Add(new DataFlow(_weights, _weightsGradient));

            return data;
        }

        public void SetWeights(double[] weights) => weights.CopyTo(_weights, 0);
    }
}
