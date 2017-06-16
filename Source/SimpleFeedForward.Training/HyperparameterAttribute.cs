//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

using System;

namespace SimpleFeedForward.Training
{
    /// <summary>
    ///     Specifies a training hyperparameter
    ///     and stores its recommended value.
    ///     
    ///     This class cannot be inherited.
    /// </summary>
    internal sealed class HyperparameterAttribute : Attribute
    {
        /// <summary>
        ///     Initializes a new instance of the HyperparameterAttribute class
        ///     with the specified recommended value.
        /// </summary>
        /// 
        /// <param name="recommendedValue">
        ///     The recommended value for the decorated hyperparameter.
        /// </param>
        public HyperparameterAttribute(double recommendedValue)
        {
            RecommendedValue = recommendedValue;
        }

        /// <summary>
        ///     Gets the recommended value.
        /// </summary>
        public double RecommendedValue { get; private set; }
    }
}
