//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

using System.Collections.Generic;
using System.Linq;

namespace SimpleFeedForward.Training
{
    /// <summary>
    ///     Represents an SGD trainer with momentum modification.
    ///     
    ///     This trainer remembers finalized gradients at each iteration,
    ///     and determines the next update as a convex combination
    ///     of the gradient and the previous update.
    /// </summary>
    /// 
    /// <remarks>
    ///     To read more about Momentum SGD trainer, use the link below.
    ///     <see cref="http://cs231n.github.io/neural-networks-3/#sgd"/>
    /// </remarks>
    public class MomentumTrainer : SgdTrainer
    {
        #region Fields

        /// <summary>
        ///     The dictionary with the updates.
        /// </summary>
        protected readonly IDictionary<double[], double[]> _updates =
               new Dictionary<double[], double[]>();

        #endregion

        #region Properties

        /// <summary>
        ///     Gets or sets the Momentum hyperparameter.
        ///     
        ///     The default value is 0.9.
        /// </summary>
        [Hyperparameter(0.9)]
        public double Momentum { get; set; } = 0.9;

        #endregion

        #region Methods

        /// <summary>
        ///     Initializes the MomentumTrainer with the specified network.
        /// </summary>
        /// 
        /// <param name="network">
        ///     The network to initialize the trainer with.
        /// </param>
        public override void Init(INetwork network)
        {
            base.Init(network);

            foreach (var gradient in network.ExtractData().Select(flow => flow.Gradient))
                _updates.Add(gradient, new double[gradient.Length]);
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

            for (int i = 0; i < gradient.Length; i++)
            {
                var velocity = Momentum * updateForGradient[i] - LearningRate * gradient[i];
                gradient[i] = velocity;
                updateForGradient[i] = velocity;
            }
        }

        #endregion
    }
}
