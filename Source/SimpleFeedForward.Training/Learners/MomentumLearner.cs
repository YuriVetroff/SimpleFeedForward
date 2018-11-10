using System.Collections.Generic;
using System.Linq;

namespace SimpleFeedForward.Training.Learners
{
    public class MomentumLearner : SgdLearner
    {
        protected readonly IDictionary<double[], double[]> _updates =
               new Dictionary<double[], double[]>();

        public override void Init(INetwork network)
        {
            base.Init(network);
            foreach (var gradient in network.ExtractData().Select(flow => flow.Gradient))
            {
                _updates.Add(gradient, new double[gradient.Length]);
            }
        }

        public double Momentum { get; set; } = 0.9;

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
    }
}
