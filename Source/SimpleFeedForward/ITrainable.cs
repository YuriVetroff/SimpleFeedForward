//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

using System.Collections.Generic;

namespace SimpleFeedForward
{
    /// <summary>
    ///     Defines a unit that contains some trainable data
    ///     in the form of values-gradient pairs.
    /// </summary>
    public interface ITrainable
    {
        /// <summary>
        ///     Extracts the trainable data of the ITrainable
        ///     as a list of data flows.
        /// </summary>
        /// 
        /// <returns>
        ///     An System.Collections.Generic.IEnumerable`1 whose elements are the set of
        ///     the trainable data.
        /// </returns>
        IEnumerable<DataFlow> ExtractData();
    }
}
