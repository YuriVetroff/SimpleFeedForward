namespace SimpleFeedForward.Layers
{
    [Activation(ActivationType.Relu)]
    public class ReluLayer : ActivationLayer
    {
        protected override double Evaluation(double x) => x > 0 ? x : 0;
        protected override double Derivative(double x) => x > 0 ? 1 : 0;
    }
}
