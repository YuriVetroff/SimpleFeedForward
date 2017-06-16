//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

using System;
using System.Collections.Generic;
using System.Linq;

namespace SimpleFeedForward.Layers
{
    /// <summary>
    ///     Defines a bias layer that modifies incoming data
    ///     shifting with its array of biases.
    /// </summary>
    public interface IBiasLayer : IWeightLayer
    {
        #region Properties

        /// <summary>
        ///     Gets the number of bias in the IBiasLayer.
        /// </summary>
        int BiasLength { get; }

        /// <summary>
        ///     Determines whether the bias is used during propagation.
        /// </summary>
        bool UseBias { get; set; }

        /// <summary>
        ///     Gets the bias of the IBiasLayer as a read-only list of double values.
        /// </summary>
        IReadOnlyList<double> Bias { get; }

        #endregion

        #region Methods

        /// <summary>
        ///     Sets the bias of the IBiasLayer with the specified array.
        /// </summary>
        /// 
        /// <param name="bias">
        ///     The array to set the bias with.
        /// </param>
        void SetBias(double[] bias);

        #endregion
    }

    /// <summary>
    ///     Represents a bias layer that modifies incoming data
    ///     shifting with its array of biases.
    ///     
    ///     This class is abstract.
    /// </summary>
    public abstract class BiasLayer : WeightLayer, IBiasLayer
    {
        #region Fields

        /// <summary>
        ///     The biases of the BiasLayer.
        ///     
        ///     In contrast to the weights, this data is one-dimensional.
        /// </summary>
        protected double[] _bias;
        /// <summary>
        ///     The array of the bias gradients that is filled during backpropagation.
        /// </summary>
        protected double[] _biasGradient;

        #endregion

        #region Properties

        /// <summary>
        ///     Gets the number of bias in the BiasLayer.
        /// </summary>
        public int BiasLength => NeuronCount;

        /// <summary>
        ///     Determines whether the bias is used during propagation.
        /// </summary>
        public bool UseBias { get; set; } = true;

        /// <summary>
        ///     Gets the bias of the BiasLayer as a read-only list of double values.
        /// </summary>
        public IReadOnlyList<double> Bias => _bias;

        #endregion

        #region Methods

        /// <summary>
        ///     Runs the BiasLayer forward, propagating data to next layers.
        /// </summary>
        public override void Forward()
        {
            base.Forward();

            if (UseBias)
                _engine.Increment(_output, _bias);
        }
        /// <summary>
        ///     Runs the BiasLayer backward, propagating error to previous layers.
        /// </summary>
        public override void Backward()
        {
            if (UseBias)
                _engine.Increment(_biasGradient, _error);

            base.Backward();
        }

        /// <summary>
        ///     Initializes the BiasLayer using the number of incoming neurons.
        /// </summary>
        /// 
        /// <param name="incomingLength">
        ///     The number of incoming neurons of the BiasLayer.
        /// </param>
        public override void Init(int incomingLength)
        {
            base.Init(incomingLength);

            _bias = new double[_outgoingLength];
            _biasGradient = new double[_outgoingLength];

            var random = new Random();
            for (int i = 0; i < _outgoingLength; i++)
                _bias[i] = random.NextDouble() / _outgoingLength;
        }

        /// <summary>
        ///     Extracts the trainable data of the BiasLayer
        ///     as a list of data flows.
        /// </summary>
        /// 
        /// <returns>
        ///     An System.Collections.Generic.IEnumerable`1 whose elements are the set of
        ///     the BiasLayer's trainable data.
        /// </returns>
        public override IEnumerable<DataFlow> ExtractData()
        {
            var data = base.ExtractData().ToList();

            if (UseBias)
                data.Add(new DataFlow(_bias, _biasGradient));

            return data;
        }

        /// <summary>
        ///     Sets the bias of the BiasLayer with the specified array.
        /// </summary>
        /// 
        /// <param name="bias">
        ///     The array to set the bias with.
        /// </param>
        public void SetBias(double[] bias) => bias.CopyTo(_bias, 0);

        #endregion
    }
}
