//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

namespace SimpleFeedForward.Training
{
    /// <summary>
    ///     Defines a trainer.
    /// </summary>
    public interface ITrainer : IInitializing<INetwork>
    {
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
        double Train(double[] input, double[] output);
    }
}
