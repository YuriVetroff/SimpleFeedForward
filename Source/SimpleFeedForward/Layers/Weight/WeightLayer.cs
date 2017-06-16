//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

using System;
using System.Collections.Generic;

namespace SimpleFeedForward.Layers
{
    /// <summary>
    ///     Defines a weighted layer that modifies incoming data
    ///     using a dot product operation with its array of weights.
    /// </summary>
    public interface IWeightLayer : ILayer, ITrainable
    {
        #region Properties

        /// <summary>
        ///     Gets the number of weights in the IWeightLayer.
        /// </summary>
        int WeightsLength { get; }
        /// <summary>
        ///     Gets the number of weights per a single neuron.
        /// </summary>
        int WeightsPerNeuron { get; }

        /// <summary>
        ///     Gets the weights of the IWeightLayer as a read-only list of double values.
        /// </summary>
        IReadOnlyList<double> Weights { get; }

        #endregion

        #region Methods

        /// <summary>
        ///     Sets the weights of the IWeightLayer with the specified array.
        /// </summary>
        /// 
        /// <param name="weights">
        ///     The array to set the weights with.
        /// </param>
        void SetWeights(double[] weights);

        #endregion
    }

    /// <summary>
    ///     Represents a weighted layer that modifies incoming data
    ///     using a dot product operation with its array of weights.
    ///     
    ///     This class is abstract.
    /// </summary>
    public abstract class WeightLayer : Layer, IWeightLayer
    {
        #region Fields

        /// <summary>
        ///     The weights of the WeightLayer.
        ///     
        ///     Logically, it's two-dimensional surface with a width that is equal to
        ///     the incoming length and a height that is equal to the neuron count.
        /// </summary>
        protected double[] _weights;
        /// <summary>
        ///     The array of the weights gradients that is filled during backpropagation.
        /// </summary>
        protected double[] _weightsGradient;

        /// <summary>
        ///     The length of the weights array.
        /// </summary>
        protected int _weightsLength;
        /// <summary>
        ///     The number of weights per a single neuron.
        /// </summary>
        protected int _weightsPerNeuron;

        #endregion

        #region Properties

        /// <summary>
        ///     Gets the number of weights in the WeightLayer.
        /// </summary>
        public int WeightsLength => _weightsLength;
        /// <summary>
        ///     Gets the number of weights per a single neuron.
        /// </summary>
        public int WeightsPerNeuron => _weightsLength / NeuronCount;

        /// <summary>
        ///     Gets the weights of the WeightLayer as a read-only list of double values.
        /// </summary>
        public IReadOnlyList<double> Weights => _weights;

        #endregion

        #region Methods

        /// <summary>
        ///     Runs the WeightLayer forward, propagating data to next layers.
        /// </summary>
        public override void Forward()
        {
            base.Forward();

            _engine.ForwardDotProduct(_input, _weights, _output);
        }
        /// <summary>
        ///     Runs the WeightLayer backward, propagating error to previous layers.
        /// </summary>
        public override void Backward()
        {
            _engine.BackwardDotProduct(_input, _backError, _weights, _weightsGradient, _error);

            base.Backward();
        }

        /// <summary>
        ///     Initializes the WeightLayer using the number of incoming neurons.
        /// </summary>
        /// 
        /// <param name="incomingLength">
        ///     The number of incoming neurons of the WeightLayer.
        /// </param>
        public override void Init(int incomingLength)
        {
            base.Init(incomingLength);

            _weightsLength = _incomingLength * _outgoingLength;
            _weightsPerNeuron = _weightsLength / NeuronCount;

            _weights = new double[_weightsLength];
            _weightsGradient = new double[_weightsLength];

            var random = new Random();
            for (int i = 0; i < _weightsLength; i++)
                _weights[i] = random.NextDouble() / _incomingLength;
        }

        /// <summary>
        ///     Extracts the trainable data of the WeightLayer
        ///     as a list of data flows.
        /// </summary>
        /// 
        /// <returns>
        ///     An System.Collections.Generic.IEnumerable`1 whose elements are the set of
        ///     the WeightLayer's trainable data.
        /// </returns>
        public virtual IEnumerable<DataFlow> ExtractData()
        {
            var data = new List<DataFlow>();

            data.Add(new DataFlow(_weights, _weightsGradient));

            return data;
        }

        /// <summary>
        ///     Sets the weights of the WeightLayer with the specified array.
        /// </summary>
        /// 
        /// <param name="weights">
        ///     The array to set the weights with.
        /// </param>
        public void SetWeights(double[] weights) => weights.CopyTo(_weights, 0);

        #endregion
    }
}
