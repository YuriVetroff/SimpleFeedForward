//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

using SimpleFeedForward.Layers;
using System.Collections.Generic;
using System.Linq;

namespace SimpleFeedForward
{
    /// <summary>
    ///     Defines a neural network.
    /// </summary>
    public interface INetwork : ITrainable, IProducing, IInitializing, ILayerSequence
    {
        /// <summary>
        ///     Runs the INetwork forward, propagating the specified input array.
        /// </summary>
        /// 
        /// <param name="input">
        ///     The input to propagate.
        /// </param>
        void Forward(double[] input);
        /// <summary>
        ///     Runs the INetwork backward, propagating the specified error array.
        /// </summary>
        /// 
        /// <param name="error">
        ///     The error to propagate.
        /// </param>
        void Backward(double[] error);
    }

    public class Network : LayerSequence, INetwork
    {
        #region Fields

        /// <summary>
        ///     The input layer of the Network.
        /// </summary>
        private InputLayer _inputLayer;

        #endregion

        #region Constructors

        /// <summary>
        ///     Initializes a new instance of the Network class
        ///     with the specified input size.
        /// </summary>
        /// 
        /// <param name="inputSize">
        ///     The input size.
        /// </param>
        public Network(int inputSize)
        {
            _inputLayer = new InputLayer(inputSize);
        }

        #endregion

        #region Properties

        /// <summary>
        ///     Gets the output of the Network.
        /// </summary>
        public IReadOnlyList<double> Output => _layers.Last().Output;

        #endregion

        #region Methods

        /// <summary>
        ///     Runs the Network forward, propagating the specified input array.
        /// </summary>
        /// 
        /// <param name="input">
        ///     The input to propagate.
        /// </param>
        public void Forward(double[] input)
        {
            _inputLayer.SetInput(input);
            _inputLayer.Forward();

            foreach (var layer in _layers)
                layer.Forward();
        }
        /// <summary>
        ///     Runs the Network backward, propagating the specified error array.
        /// </summary>
        /// 
        /// <param name="error">
        ///     The error to propagate.
        /// </param>
        public void Backward(double[] error)
        {
            _layers.Last().SetError(error);
            foreach (var layer in _layers.Reverse())
                layer.Backward();
        }

        /// <summary>
        ///     Initializes the Network and all its components.
        /// </summary>
        public void Init()
        {
            _layers[0].Init(_inputLayer);
            for (int i = 1; i < _layers.Count; i++)
                _layers[i].Init(_layers[i - 1]);
        }

        /// <summary>
        ///     Extracts the trainable data of the Network
        ///     as a list of data flows.
        /// </summary>
        /// 
        /// <returns>
        ///     An System.Collections.Generic.IEnumerable`1 whose elements are the set of
        ///     the Network's trainable data.
        /// </returns>
        public IEnumerable<DataFlow> ExtractData()
        {
            var data = new List<DataFlow>();

            foreach (ITrainable weightLayer in _layers.Where(layer => layer is ITrainable))
                data.AddRange(weightLayer.ExtractData());

            return data;
        }

        #endregion
    }
}
