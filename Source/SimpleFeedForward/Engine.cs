using System;

namespace SimpleFeedForward
{
    public class Engine
    {
        #region Singleton

        private static volatile Engine _instance;
        private static object _syncRoot = new object();

        private Engine() { }

        public static Engine Instance
        {
            get
            {
                if (_instance == null)
                {
                    lock (_syncRoot)
                    {
                        if (_instance == null)
                        {
                            _instance = new Engine();
                        }
                    }
                }

                return _instance;
            }
        }

        #endregion

        #region Arithmetic

        public void Increment(double[] destination, double[] increment)
        {
            var length = destination.Length;
            if (length != increment.Length)
            {
                throw new ArgumentException(
                   "Cannot perform the increment operation to arrays with different sizes!");
            }

            for (int index = 0; index < length; index++)
            {
                destination[index] += increment[index];
            }
        }

        #endregion

        #region Propagation

        public void ForwardDotProduct(double[] input, double[] weights, double[] output)
        {
            var incomingLength = input.Length;
            var outgoingLength = output.Length;

            for (int neuronIndex = 0; neuronIndex < outgoingLength; neuronIndex++)
            {
                var sum = 0.0d;
                for (int weightIndex = 0; weightIndex < incomingLength; weightIndex++)
                {
                    sum += input[weightIndex] * weights[weightIndex + neuronIndex * incomingLength];
                }

                output[neuronIndex] = sum;
            }
        }
        public void BackwardDotProduct(double[] input, double[] backError,
            double[] weights, double[] weightsGradient, double[] error)
        {
            var incomingLength = input.Length;
            var outgoingLength = error.Length;

            Array.Clear(backError, 0, incomingLength);
            for (int neuronIndex = 0; neuronIndex < outgoingLength; neuronIndex++)
            {
                var currentError = error[neuronIndex];
                for (int weightIndex = 0; weightIndex < incomingLength; weightIndex++)
                {
                    var baseIndex = neuronIndex * incomingLength;
                    weightsGradient[weightIndex + baseIndex] += input[weightIndex] * currentError;
                    backError[weightIndex] += weights[weightIndex + baseIndex] * currentError;
                }
            }
        }

        public void ForwardActivation(double[] input, double[] output,
            Func<double, double> evaluation)
        {
            var incomingLength = input.Length;
            for (int index = 0; index < incomingLength; index++)
            {
                output[index] = evaluation(input[index]);
            }
        }
        public void BackwardActivation(double[] input, double[] backError,
            double[] error, Func<double, double> derivative)
        {
            var incomingLength = input.Length;
            for (int index = 0; index < incomingLength; index++)
            {
                backError[index] = derivative(input[index]) * error[index];
            }
        }

        #endregion
    }
}
