//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

namespace SimpleFeedForward.Layers
{
    /// <summary>
    ///     Represents an activation layer that applies an activation function
    ///     to incoming data, prodiving nonlinearity.
    ///     
    ///     This class is abstract.
    /// </summary>
    public abstract class ActivationLayer : Layer
    {
        #region Methods

        /// <summary>
        ///     Runs the ActivationLayer forward, propagating data to next layers.
        /// </summary>
        public override void Forward()
        {
            base.Forward();

            _engine.ForwardActivation(_input, _output, Evaluation);
        }
        /// <summary>
        ///     Runs the ActivationLayer backward, propagating error to previous layers.
        /// </summary>
        public override void Backward()
        {
            _engine.BackwardActivation(_input, _backError, _error, Derivative);

            base.Backward();
        }

        /// <summary>
        ///     Initializes the ActivationLayer, connecting it to the incoming (previous) layer.
        /// </summary>
        /// 
        /// <param name="incoming">
        ///     The previous layer of the ActivationLayer.
        /// </param>
        public override void Init(int incomingLength)
        {
            base.Init(incomingLength);

            _outgoingLength = incomingLength;
            _output = new double[_outgoingLength];
            _error = new double[_outgoingLength];
        }
        
        /// <summary>
        ///     Applies the activation to a value.
        ///     
        ///     Used in forward propagation.
        /// </summary>
        /// 
        /// <param name="x">
        ///     The value.
        /// </param>
        /// 
        /// <returns>
        ///     The result of the activation.
        /// </returns>
        protected abstract double Evaluation(double x);
        /// <summary>
        ///     Gets the activation derivative for a value.
        ///     
        ///     Used in backward propagation.
        /// </summary>
        /// 
        /// <param name="x">
        ///     The value.
        /// </param>
        /// 
        /// <returns>
        ///     The activation derivative.
        /// </returns>
        protected abstract double Derivative(double x);

        #endregion
    }
}
