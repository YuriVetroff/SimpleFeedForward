namespace SimpleFeedForward.Layers
{
    public abstract class ActivationLayer : Layer
    {
        public override void Forward()
        {
            base.Forward();

            _engine.ForwardActivation(_input, _output, Evaluation);
        }
        public override void Backward()
        {
            _engine.BackwardActivation(_input, _backError, _error, Derivative);

            base.Backward();
        }

        public override void Init(int incomingLength)
        {
            base.Init(incomingLength);

            _outgoingLength = incomingLength;
            _output = new double[_outgoingLength];
            _error = new double[_outgoingLength];
        }
        
        protected abstract double Evaluation(double x);
        protected abstract double Derivative(double x);
    }
}
