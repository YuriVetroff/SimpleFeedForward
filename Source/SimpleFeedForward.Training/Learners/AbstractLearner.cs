namespace SimpleFeedForward.Training.Learners
{
    public abstract class AbstractLearner : ILearner
    {
        protected INetwork _network;

        public virtual void Init(INetwork parameter)
        {
            _network = parameter;
        }

        public double LearningRate { get; set; } = 0.001;

        public void UpdateWeights()
        {
            var dataFlows = _network.ExtractData();

            foreach (var flow in dataFlows)
            {
                var weights = flow.Values;
                var gradient = flow.Gradient;

                FinalizeGradient(gradient);

                flow.Commit();
            }
        }

        protected abstract void FinalizeGradient(double[] gradient);
    }
}
