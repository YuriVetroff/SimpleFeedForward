//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

using System;
using System.Collections.Generic;

namespace SimpleFeedForward.Layers
{
    /// <summary>
    ///     Represents an input layer.
    ///     
    ///     The main feature of the InputLayer, which distinguishes it from other layers,
    ///     is that it uses its input data as the output without any evaluation.
    ///     
    ///     This class is internal because it is created inside a network
    ///     and does not have to be visible outside the library.
    /// </summary>
    internal class InputLayer : Layer
    {
        #region Constructors

        /// <summary>
        ///     Initializes a new instance of the InputLayer class
        ///     with the specified number of neurons.
        /// </summary>
        /// 
        /// <param name="neuronCount">
        ///     The number of neurons in the InputLayer.
        /// </param>
        public InputLayer(int neuronCount)
        {
            Init(neuronCount);
        }

        #endregion

        #region Properties

        /// <summary>
        ///     Gets the output of the InputLayer as a read-only list of double values.
        ///     
        ///     Actually, this is the input data given to the layer.
        /// </summary>
        public override IReadOnlyList<double> Output => _input;

        #endregion

        #region Methods

        /// <summary>
        ///     Runs the InputLayer forward, propagating data to next layers.
        ///     
        ///     In case of InputLayer, it is just a replication of the input to the output.
        /// </summary>
        public override void Forward()
        {
            _input.CopyTo(_output, 0);
        }

        /// <summary>
        ///     Initializes the InputLayer, connecting it to the incoming (previous) layer.
        ///     
        ///     An input layer cannot be connected to another (previous) layer.
        ///     This operation is invalid and leads to an exception.
        /// </summary>
        /// 
        /// <param name="incoming">
        ///     The previous layer of the Layer.
        /// </param>
        /// 
        /// <exception cref="System.InvalidOperationException">
        ///     An attempt to connect the InputLayer to another layer.
        /// </exception>
        public sealed override void Init(ILayer incoming)
        {
            throw new InvalidOperationException(
                "Cannot initialize an input layer with an incoming layer!");
        }
        /// <summary>
        ///     Initializes the InputLayer using the number of incoming neurons.
        ///     
        ///     In case of InputLayer, the number of incoming neurons equals
        ///     the number of outgoing neurons (the neuron count).
        /// </summary>
        /// 
        /// <param name="incomingLength">
        ///     The number of incoming neurons of the InputLayer.
        /// </param>
        public sealed override void Init(int incomingLength)
        {
            base.Init(incomingLength);

            _outgoingLength = incomingLength;

            _output = new double[_outgoingLength];
            _error = new double[_outgoingLength];
        }

        #endregion
    }
}
