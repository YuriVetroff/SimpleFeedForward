using System;
using System.Collections.Generic;
using System.Linq;

namespace SimpleFeedForward.Training.Learners
{
    public class SimpleAdamLearner : AbstractLearner
    {
        protected readonly IDictionary<double[], double[]> _updates =
            new Dictionary<double[], double[]>();
        protected readonly IDictionary<double[], double[]> _secondUpdates =
            new Dictionary<double[], double[]>();

        public override void Init(INetwork network)
        {
            base.Init(network);
            foreach (var gradient in network.ExtractData().Select(flow => flow.Gradient))
            {
                _updates.Add(gradient, new double[gradient.Length]);
                _secondUpdates.Add(gradient, new double[gradient.Length]);
            }
        }

        public double Beta1 { get; set; } = 0.9;
        public double Beta2 { get; set; } = 0.999;
        public double Eps { get; set; } = 1e-8;

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

        protected virtual double ComputeDelta(double m, double v)
            => -LearningRate * (m / (Math.Sqrt(v) + Eps));
    }
}
