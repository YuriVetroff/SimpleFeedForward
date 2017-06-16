//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

using System;

namespace SimpleFeedForward.Layers
{
    /// <summary>
    ///     Represents an activation layer that uses the hyperbolic tangent function
    ///     to provide nonlinearity.
    /// </summary>
    [Activation(ActivationType.Tanh)]
    public class TanhLayer : ActivationLayer
    {
        #region Methods

        /// <summary>
        ///     Applies the tanh function to a value.
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
        protected override double Evaluation(double x) => Math.Tanh(x);
        /// <summary>
        ///     Gets the tanh function derivative for a value.
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
        protected override double Derivative(double x)
        {
            var y = Evaluation(x);
            return 1 - y * y;
        }

        #endregion
    }
}
