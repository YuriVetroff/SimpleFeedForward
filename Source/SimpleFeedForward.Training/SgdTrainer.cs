//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

namespace SimpleFeedForward.Training
{
    /// <summary>
    ///     Represents a classic stochastic gradient descent (SGD) trainer.
    /// </summary>
    /// 
    /// <remarks>
    ///     To read more about SGD trainers, use the link below.
    ///     <see cref="http://cs231n.github.io/neural-networks-3/#sgd"/>
    /// </remarks>
    public class SgdTrainer : ITrainer
    {
        #region Fields

        /// <summary>
        ///     The instance of a network that is going to be trained.
        /// </summary>
        protected INetwork _network;

        #endregion

        #region Constructors

        /// <summary>
        ///     Initializes a new instance of the SgdTrainer class.
        /// </summary>
        public SgdTrainer() { }

        #endregion

        #region Properties

        /// <summary>
        ///     Gets or sets the learning rate hyperparameter.
        ///     
        ///     The default value is 0.001.
        /// </summary>
        [Hyperparameter(0.001)]
        public double LearningRate { get; set; } = 0.001;

        #endregion

        #region Methods

        /// <summary>
        ///     Initializes the SgdTrainer with the specified network.
        /// </summary>
        /// 
        /// <param name="network">
        ///     The network to initialize the trainer with.
        /// </param>
        public virtual void Init(INetwork network)
        {
            _network = network;
        }

        /// <summary>
        ///     Performs a single training iteration using the specified training data.
        ///     Returns the training loss.
        /// </summary>
        /// 
        /// <param name="input">
        ///     The input data that will be passed through the network.
        /// </param>
        /// 
        /// <param name="output">
        ///     The output data that is expected for the passed input.
        /// </param>
        /// 
        /// <returns>
        ///     The loss between the expected and actual outputs.
        /// </returns>
        public double Train(double[] input, double[] output)
        {
            _network.Forward(input);
            var actualOutput = _network.Output;
            var outputLength = output.Length;
            var distance = new double[outputLength];
            for (int i = 0; i < outputLength; i++)
                distance[i] = actualOutput[i] - output[i];

            var loss = CalculateLoss(distance);

            _network.Backward(distance);
            UpdateWeights();

            return loss;
        }

        /// <summary>
        ///     Updates the weights in the network.
        /// </summary>
        protected virtual void UpdateWeights()
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

        /// <summary>
        ///     Improves a gradient vector.
        /// </summary>
        /// 
        /// <param name="gradient">
        ///     The gradient vector to improve.
        /// </param>
        protected virtual void FinalizeGradient(double[] gradient)
        {
            for (int i = 0; i < gradient.Length; i++)
                gradient[i] = -gradient[i] * LearningRate;
        }

        /// <summary>
        ///     Calculates loss using a distance between expected and actual outputs.
        /// </summary>
        /// 
        /// <param name="distance">
        ///     The distance between expected and actual outputs.
        /// </param>
        /// 
        /// <returns>
        ///     The loss.
        /// </returns>
        protected virtual double CalculateLoss(double[] distance)
        {
            var loss = 0.0d;
            foreach (var item in distance)
                loss += item * item;
            loss /= 2;

            return loss;
        }

        #endregion
    }
}
