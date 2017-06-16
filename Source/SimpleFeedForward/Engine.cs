//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

using System;

namespace SimpleFeedForward
{
    /// <summary>
    ///     Represents a computation engine that provides mechanics
    ///     of forward and backward propagation.
    ///     
    ///     This class is a singleton.
    /// </summary>
    public class Engine
    {
        #region Singleton

        /// <summary>
        ///     The instance of the singleton.
        /// </summary>
        private static volatile Engine _instance;
        /// <summary>
        ///     The synchronization root to provide access from different threads.
        /// </summary>
        private static object _syncRoot = new object();

        /// <summary>
        ///     Initializes a new instance of the Engine class.
        ///     
        ///     Note that the constructor is accessible only inside the class.
        /// </summary>
        private Engine() { }

        /// <summary>
        ///     Gets the actual instance of the Engine.
        ///     
        ///     Actually, it is just a singleton getter
        ///     with a double-check locking mechanism.
        /// </summary>
        public static Engine Instance
        {
            get
            {
                if (_instance == null)
                {
                    lock (_syncRoot)
                    {
                        if (_instance == null)
                            _instance = new Engine();
                    }
                }

                return _instance;
            }
        }

        #endregion

        #region Arithmetic

        /// <summary>
        ///     Increments the values of the first operand
        ///     with the values from the seconds operand.
        /// </summary>
        /// 
        /// <param name="destination">
        ///     The destination array which values is going to be incremented.
        /// </param>
        /// 
        /// <param name="increment">
        ///     The increment array which values is going to be added to the destination array.
        /// </param>
        /// 
        /// <exception cref="System.ArgumentException">
        ///     The arrays have different sizes.
        /// </exception>
        public void Increment(double[] destination, double[] increment)
        {
            var length = destination.Length;
            if (length != increment.Length)
                throw new ArgumentException(
                    "Cannot perform the increment operation to arrays with different sizes!");

            for (int index = 0; index < length; index++)
                destination[index] += increment[index];
        }

        #endregion

        #region Propagation

        /// <summary>
        ///     Performs the forward propagation using dot product mechanics.
        /// </summary>
        /// 
        /// <param name="input">
        ///     The input array with the initial data.
        /// </param>
        /// 
        /// <param name="weights">
        ///     The weights array whose subarrays are multiplied by the input array.
        /// </param>
        /// 
        /// <param name="output">
        ///     The output array where the result is stored.
        /// </param>
        public void ForwardDotProduct(double[] input, double[] weights, double[] output)
        {
            var incomingLength = input.Length;
            var outgoingLength = output.Length;

            for (int neuronIndex = 0; neuronIndex < outgoingLength; neuronIndex++)
            {
                var sum = 0.0d;
                for (int weightIndex = 0; weightIndex < incomingLength; weightIndex++)
                    sum += input[weightIndex] * weights[weightIndex + neuronIndex * incomingLength];

                output[neuronIndex] = sum;
            }
        }

        /// <summary>
        ///     Performs the backward propagation using dot product mechanics.
        /// </summary>
        /// 
        /// <param name="input">
        ///     The input array with the initial data.
        /// </param>
        /// 
        /// <param name="backError">
        ///     The array where the backpropagated error is stored.
        /// </param>
        /// 
        /// <param name="weights">
        ///     The weights array whose subarrays are multiplied by the input array.
        /// </param>
        /// 
        /// <param name="weightsGradient">
        ///     The array where the weights gradient is stored.
        /// </param>
        /// 
        /// <param name="error">
        ///     The array with the received error data.
        /// </param>
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

        /// <summary>
        ///     Performs the forward propagation using activation function mechanics.
        /// </summary>
        /// 
        /// <param name="input">
        ///     The input array with the initial data.
        /// </param>
        /// 
        /// <param name="output">
        ///     The output array where the result is stored.
        /// </param>
        /// 
        /// <param name="evaluation">
        ///     The activation evaluation.
        /// </param>
        public void ForwardActivation(double[] input, double[] output,
            Func<double, double> evaluation)
        {
            var incomingLength = input.Length;
            for (int index = 0; index < incomingLength; index++)
                output[index] = evaluation(input[index]);
        }

        /// <summary>
        ///     Performs the backward propagation using activation function mechanics.
        /// </summary>
        /// 
        /// <param name="input">
        ///     The input array with the initial data.
        /// </param>
        /// 
        /// <param name="backError">
        ///     The array where the backpropagated error is stored.
        /// </param>
        /// 
        /// <param name="error">
        ///     The array with the received error data.
        /// </param>
        /// 
        /// <param name="derivative">
        ///     The activation derivative.
        /// </param>
        public void BackwardActivation(double[] input, double[] backError,
            double[] error, Func<double, double> derivative)
        {
            var incomingLength = input.Length;
            for (int index = 0; index < incomingLength; index++)
                backError[index] = derivative(input[index]) * error[index];
        }

        #endregion
    }
}
