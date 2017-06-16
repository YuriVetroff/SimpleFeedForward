//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

using System.Collections.Generic;
using System.Linq;

namespace SimpleFeedForward.Layers
{
    /// <summary>
    ///     Defines a layer of a neural network.
    ///     
    ///     A layer is a structural node of a network.
    ///     The model contains general information about a layer
    ///     and provides methods for forward and backward propagation.
    /// </summary>
    public interface ILayer : IInitializing<ILayer>, IInitializing<int>, IProducing
    {
        #region Properties

        /// <summary>
        ///     Gets the number of neurons in the ILayer.
        /// </summary>
        int NeuronCount { get; }

        /// <summary>
        ///     Gets the error of the ILayer as a read-only list of double values.
        /// </summary>
        IReadOnlyList<double> Error { get; }

        #endregion

        #region Methods

        /// <summary>
        ///     Runs the ILayer forward, propagating data to next layers.
        /// </summary>
        void Forward();
        /// <summary>
        ///     Runs the ILayer backward, propagating error to previous layers.
        /// </summary>
        void Backward();

        /// <summary>
        ///     Sets the input of the ILayer with the specified array.
        /// </summary>
        /// 
        /// <param name="input">
        ///     The array to set the input with.
        /// </param>
        void SetInput(double[] input);
        /// <summary>
        ///     Sets the error of the ILayer with the specified array.
        /// </summary>
        /// 
        /// <param name="error">
        ///     The array to set the error with.
        /// </param>
        void SetError(double[] error);

        #endregion
    }

    /// <summary>
    ///     Represents a layer of a neural network.
    ///     This class is abstract.
    /// </summary>
    public abstract class Layer : ILayer
    {
        #region Fields

        /// <summary>
        ///     The instance of the engine that performs propagation mechanics.
        /// </summary>
        protected readonly Engine _engine = Engine.Instance;

        /// <summary>
        ///     The incoming length of the Layer.
        ///     
        ///     This value describes the number of neurons in the previous layer (if it exists)
        ///     or just the shape of the Layer's input data.
        /// </summary>
        protected int _incomingLength;
        /// <summary>
        ///     The outgoing length of the Layer.
        ///     
        ///     In other words, this is the neuron count.
        /// </summary>
        protected int _outgoingLength;

        /// <summary>
        ///     The input buffer of the Layer.
        ///     
        ///     Contains the last input data received by the Layer.
        /// </summary>
        protected double[] _input;
        /// <summary>
        ///     The output buffer of the Layer.
        ///     
        ///     Contains the last input data evaluated by the Layer
        ///     and written to the output.
        /// </summary>
        protected double[] _output;

        /// <summary>
        ///     The error buffer of the Layer.
        ///     
        ///     Contains the error vector received from next layers
        ///     during backpropagation.
        /// </summary>
        protected double[] _error;
        /// <summary>
        ///     The back error buffer of the Layer.
        ///     
        ///     Actually, this is a temporary array with error vector,
        ///     that has been already processed by a backpropagation method
        ///     but has not been sent to the previous layer.
        /// </summary>
        protected double[] _backError;

        /// <summary>
        ///     The previous layer of the Layer.
        /// </summary>
        protected ILayer _incoming;

        #endregion

        #region Constructors

        /// <summary>
        ///     Initializes a new instance of the Layer class.
        /// </summary>
        public Layer()
        { }

        #endregion

        #region Properties

        /// <summary>
        ///     Gets the number of neurons in the Layer.
        /// </summary>
        public int NeuronCount => _outgoingLength;

        /// <summary>
        ///     Gets the output of the Layer as a read-only list of double values.
        /// </summary>
        public virtual IReadOnlyList<double> Output => _output;
        /// <summary>
        ///     Gets the error of the Layer as a read-only list of double values.
        /// </summary>
        public IReadOnlyList<double> Error => _error;

        #endregion

        #region Methods

        /// <summary>
        ///     Runs the Layer forward, propagating data to next layers.
        /// </summary>
        public virtual void Forward()
        {
            if (_incoming != null)
                SetInput(_incoming.Output.ToArray());
        }
        /// <summary>
        ///     Runs the Layer backward, propagating error to previous layers.
        /// </summary>
        public virtual void Backward()
        {
            if (_incoming != null)
                _incoming.SetError(_backError);
        }

        /// <summary>
        ///     Initializes the Layer, connecting it to the incoming (previous) layer.
        /// </summary>
        /// 
        /// <param name="incoming">
        ///     The previous layer of the Layer.
        /// </param>
        public virtual void Init(ILayer incoming)
        {
            _incoming = incoming;
            Init(incoming.NeuronCount);
        }
        /// <summary>
        ///     Initializes the Layer using the number of incoming neurons.
        /// </summary>
        /// 
        /// <param name="incomingLength">
        ///     The number of incoming neurons of the Layer.
        /// </param>
        public virtual void Init(int incomingLength)
        {
            _incomingLength = incomingLength;
            _input = new double[incomingLength];
            _backError = new double[incomingLength];
        }

        /// <summary>
        ///     Sets the input of the Layer with the specified array.
        /// </summary>
        /// 
        /// <param name="input">
        ///     The array to set the input with.
        /// </param>
        public void SetInput(double[] input) => input.CopyTo(_input, 0);
        /// <summary>
        ///     Sets the error of the Layer with the specified array.
        /// </summary>
        /// 
        /// <param name="error">
        ///     The array to set the error with.
        /// </param>
        public void SetError(double[] error) => error.CopyTo(_error, 0);

        #endregion
    }
}
