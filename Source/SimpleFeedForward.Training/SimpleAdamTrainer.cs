//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

using System;
using System.Collections.Generic;
using System.Linq;

namespace SimpleFeedForward.Training
{
    /// <summary>
    ///     Represents an adaptive moment estimation (Adam) trainer
    ///     at its simplified notation.
    ///     
    ///     In this trainer, running averages of both the gradients
    ///     and the second moments of the gradients are used.
    /// </summary>
    /// 
    /// <remarks>
    ///     To read more about Adam trainer, use the link below.
    ///     <see cref="http://cs231n.github.io/neural-networks-3/#ada"/>
    /// </remarks>
    public class SimpleAdamTrainer : SgdTrainer
    {
        #region Fields

        /// <summary>
        ///     The dictionary with the updates.
        /// </summary>
        protected readonly IDictionary<double[], double[]> _updates =
            new Dictionary<double[], double[]>();
        /// <summary>
        ///     The dictionary with the second updates.
        /// </summary>
        protected readonly IDictionary<double[], double[]> _secondUpdates =
            new Dictionary<double[], double[]>();

        #endregion

        #region Properties

        /// <summary>
        ///     Gets or sets the Beta1 hyperparameter.
        ///     
        ///     The default value is 0.9.
        /// </summary>
        [Hyperparameter(0.9)]
        public double Beta1 { get; set; } = 0.9;
        /// <summary>
        ///     Gets or sets the Beta2 hyperparameter.
        ///     
        ///     The default value is 0.999.
        /// </summary>
        [Hyperparameter(0.999)]
        public double Beta2 { get; set; } = 0.999;
        /// <summary>
        ///     Gets or sets the Eps hyperparameter.
        ///     
        ///     The default value is 1e-8.
        /// </summary>
        [Hyperparameter(1e-8)]
        public double Eps { get; set; } = 1e-8;

        #endregion

        #region Methods

        /// <summary>
        ///     Initializes the SimpleAdamTrainer with the specified network.
        /// </summary>
        /// 
        /// <param name="network">
        ///     The network to initialize the trainer with.
        /// </param>
        public override void Init(INetwork network)
        {
            base.Init(network);

            foreach (var gradient in network.ExtractData().Select(flow => flow.Gradient))
            {
                _updates.Add(gradient, new double[gradient.Length]);
                _secondUpdates.Add(gradient, new double[gradient.Length]);
            }
        }

        /// <summary>
        ///     Improves a gradient vector.
        /// </summary>
        /// 
        /// <param name="gradient">
        ///     The gradient vector to improve.
        /// </param>
        protected override void FinalizeGradient(double[] gradient)
        {
            var updateForGradient = _updates[gradient];
            var secondUpdateForGradient = _updates[gradient];

            for (int i = 0; i < gradient.Length; i++)
            {
                var delta = gradient[i];
                var m = Beta1 * updateForGradient[i] + (1 - Beta1) * delta;
                var v = Beta2 * secondUpdateForGradient[i] + (1 - Beta2) * Math.Pow(delta, 2);

                gradient[i] = ComputeDelta(m, v);

                updateForGradient[i] = m;
                secondUpdateForGradient[i] = v;
            }
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
        protected virtual double ComputeDelta(double m, double v)
        {
            return -LearningRate * (m / (Math.Sqrt(v) + Eps));
        }

        #endregion
    }
}
