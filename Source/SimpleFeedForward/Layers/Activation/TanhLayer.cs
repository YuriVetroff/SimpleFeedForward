using System;

namespace SimpleFeedForward.Layers
{
    [Activation(ActivationType.Tanh)]
    public class TanhLayer : ActivationLayer
    {
        protected override double Evaluation(double x) => Math.Tanh(x);
        protected override double Derivative(double x)
        {
            var y = Evaluation(x);
            return 1 - y * y;
        }
    }
}
