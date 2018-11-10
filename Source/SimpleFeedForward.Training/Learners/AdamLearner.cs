using System;

namespace SimpleFeedForward.Training.Learners
{
    public class AdamLearner : SimpleAdamLearner
    {
        protected int _counter = 1;

        protected override void FinalizeGradient(double[] gradient)
        {
            base.FinalizeGradient(gradient);
            _counter++;
        }

        protected override double ComputeDelta(double m, double v)
        {
            var mt = m / (1 - Math.Pow(Beta1, _counter));
            var vt = v / (1 - Math.Pow(Beta2, _counter));

            return base.ComputeDelta(mt, vt);
        }
    }
}
