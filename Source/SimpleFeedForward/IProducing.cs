//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

using System.Collections.Generic;

namespace SimpleFeedForward
{
    /// <summary>
    ///     Defines a model that can produce data
    ///     and store the product in some output buffer.
    /// </summary>
    public interface IProducing
    {
        /// <summary>
        ///     Gets the output of the IProducing.
        /// </summary>
        IReadOnlyList<double> Output { get; }
    }
}
