using System;
using System.Collections.Generic;
using System.Linq;

namespace SimpleFeedForward.Layers
{
    public interface IBiasLayer : IWeightLayer
    {
        int BiasLength { get; }
        bool UseBias { get; set; }

        IReadOnlyList<double> Bias { get; }

        void SetBias(double[] bias);
    }

    public abstract class BiasLayer : WeightLayer, IBiasLayer
    {
        protected double[] _bias;
        protected double[] _biasGradient;

        public int BiasLength => NeuronCount;
        public bool UseBias { get; set; } = true;

        public IReadOnlyList<double> Bias => _bias;

        public override void Forward()
        {
            base.Forward();

            if (UseBias)
            {
                _engine.Increment(_output, _bias);
            }
        }
        public override void Backward()
        {
            if (UseBias)
            {
                _engine.Increment(_biasGradient, _error);
            }

            base.Backward();
        }

        public override void Init(int incomingLength)
        {
            base.Init(incomingLength);

            _bias = new double[_outgoingLength];
            _biasGradient = new double[_outgoingLength];

            var random = new Random();
            for (int i = 0; i < _outgoingLength; i++)
            {
                _bias[i] = random.NextDouble() / _outgoingLength;
            }
        }

        public override IEnumerable<DataFlow> ExtractData()
        {
            var data = base.ExtractData().ToList();

            if (UseBias)
            {
                data.Add(new DataFlow(_bias, _biasGradient));
            }

            return data;
        }

        public void SetBias(double[] bias) => bias.CopyTo(_bias, 0);
    }
}
