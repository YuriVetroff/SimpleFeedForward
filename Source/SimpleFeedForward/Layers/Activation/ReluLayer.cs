//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

namespace SimpleFeedForward.Layers
{
    /// <summary>
    ///     Represents an activation layer that uses the rectified linear unit
    ///     to provide nonlinearity.
    /// </summary>
    [Activation(ActivationType.Relu)]
    public class ReluLayer : ActivationLayer
    {
        #region Methods

        /// <summary>
        ///     Applies the ReLU to a value.
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
        protected override double Evaluation(double x) => x > 0 ? x : 0;
        /// <summary>
        ///     Gets the ReLU derivative for a value.
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
        protected override double Derivative(double x) => x > 0 ? 1 : 0;

        #endregion
    }
}
