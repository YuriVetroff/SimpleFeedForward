using System;
using System.Collections.Generic;

namespace SimpleFeedForward
{
    public class DataFlow
    {
        private readonly double[] _values;

        public DataFlow(double[] values, double[] gradient)
        {
            if (values.Length != gradient.Length)
            {
                throw new ArgumentException(
                    "Cannot initialize a flow with different sizes of values and gradients!");
            }

            _values = values;
            Gradient = gradient;
        }

        public IReadOnlyList<double> Values => _values;
        public double[] Gradient { get; }

        public void Commit()
        {
            for (int i = 0; i < _values.Length; i++)
            {
                _values[i] += Gradient[i];
                Gradient[i] = 0;
            }
        }
    }
}
