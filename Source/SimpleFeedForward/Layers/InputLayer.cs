using System;
using System.Collections.Generic;

namespace SimpleFeedForward.Layers
{
    internal class InputLayer : Layer
    {
        public InputLayer(int neuronCount)
        {
            Init(neuronCount);
        }

        public override IReadOnlyList<double> Output => _input;

        public override void Forward() => _input.CopyTo(_output, 0);

        public sealed override void Init(ILayer incoming)
        {
            throw new InvalidOperationException(
                "Cannot initialize an input layer with an incoming layer!");
        }
        public sealed override void Init(int incomingLength)
        {
            base.Init(incomingLength);

            _outgoingLength = incomingLength;

            _output = new double[_outgoingLength];
            _error = new double[_outgoingLength];
        }
    }
}
