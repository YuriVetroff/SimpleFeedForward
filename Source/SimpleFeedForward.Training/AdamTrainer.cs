//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

using System;

namespace SimpleFeedForward.Training
{
    /// <summary>
    ///     Represents an adaptive moment estimation (Adam) trainer
    ///     at its full notation.
    ///     
    ///     In this trainer, running averages of both the gradients
    ///     and the second moments of the gradients are used.
    /// </summary>
    /// 
    /// <remarks>
    ///     To read more about Adam trainer, use the link below.
    ///     <see cref="http://cs231n.github.io/neural-networks-3/#ada"/>
    /// </remarks>
    public class AdamTrainer : SimpleAdamTrainer
    {
        #region Fields

        /// <summary>
        ///     The iteration counter used in delta computations.
        /// </summary>
        protected int _counter = 1;

        #endregion

        #region Methods

        /// <summary>
        ///     Improves a gradient vector.
        /// </summary>
        /// 
        /// <param name="gradient">
        ///     The gradient vector to improve.
        /// </param>
        protected override void FinalizeGradient(double[] gradient)
        {
            base.FinalizeGradient(gradient);
            _counter++;
        }

        /// <summary>
        ///     Computes the final delta using the m and v values.
        /// </summary>
        /// 
        /// <param name="m">
        ///     The m value.
        /// </param>
        /// 
        /// <param name="v">
        ///     The v value.
        /// </param>
        /// 
        /// <returns>
        ///     The final delta.
        /// </returns>
        protected override double ComputeDelta(double m, double v)
        {
            var mt = m / (1 - Math.Pow(Beta1, _counter));
            var vt = v / (1 - Math.Pow(Beta2, _counter));

            return base.ComputeDelta(mt, vt);
        }

        #endregion
    }
}
