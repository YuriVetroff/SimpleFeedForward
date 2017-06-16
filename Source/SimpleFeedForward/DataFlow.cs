//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

using System;
using System.Collections.Generic;

namespace SimpleFeedForward
{
    /// <summary>
    ///     Represents a flow of neural trainable data
    ///     with the information of current values and its gradient.
    /// </summary>
    public class DataFlow
    {
        #region Fields

        /// <summary>
        ///     The values.
        /// </summary>
        private readonly double[] _values;
        /// <summary>
        ///     The gradients.
        /// </summary>
        private readonly double[] _gradient;

        #endregion

        #region Constructors

        /// <summary>
        ///     Initializes a new instance of the DataFlow class
        ///     with the specified values and gradient arrays.
        /// </summary>
        /// 
        /// <param name="values">
        ///     The array with the values.
        /// </param>
        /// 
        /// <param name="gradient">
        ///     The array with the gradient.
        /// </param>
        public DataFlow(double[] values, double[] gradient)
        {
            if (values.Length != gradient.Length)
                throw new ArgumentException("Cannot initialize a flow with different sizes of values and gradients!");

            _values = values;
            _gradient = gradient;
        }

        #endregion

        #region Properties

        /// <summary>
        ///     Gets the values as a read-only list.
        /// </summary>
        public IReadOnlyList<double> Values => _values;
        /// <summary>
        ///     Gets the gradient.
        /// </summary>
        public double[] Gradient => _gradient;

        #endregion

        #region Methods

        /// <summary>
        ///     Commits the flow, applying the gradient to the values.
        /// </summary>
        public void Commit()
        {
            for (int i = 0; i < _values.Length; i++)
            {
                _values[i] += _gradient[i];
                _gradient[i] = 0;
            }
        }

        #endregion
    }
}
