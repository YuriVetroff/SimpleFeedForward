//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

namespace SimpleFeedForward.Layers
{
    /// <summary>
    ///     Represents a usual fully-connected layer
    ///     that evaluates data using the "each-to-each" mechanics.
    /// </summary>
    public class FullyConnectedLayer : BiasLayer
    {
        /// <summary>
        ///     Initializes a new instance of the FullyConnectedLayer class
        ///     with the specified number of neurons.
        /// </summary>
        /// 
        /// <param name="neuronCount">
        ///     The number of neurons in the FullyConnectedLayer.
        /// </param>
        public FullyConnectedLayer(int neuronCount)
        {
            _outgoingLength = neuronCount;

            _output = new double[neuronCount];
            _error = new double[neuronCount];
        }
    }
}
