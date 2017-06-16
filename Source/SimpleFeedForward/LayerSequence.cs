//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

using SimpleFeedForward.Layers;
using System.Collections.Generic;

namespace SimpleFeedForward
{
    /// <summary>
    ///     Defines a sequence of layers.
    /// </summary>
    public interface ILayerSequence
    {
        /// <summary>
        ///     Adds a layer to the ILayerSequence.
        /// </summary>
        /// 
        /// <param name="layer">
        ///     The layer to add.
        /// </param>
        void Add(ILayer layer);
    }

    /// <summary>
    ///     Represents a sequence of layers.
    ///     
    ///     This class is abstract.
    /// </summary>
    public abstract class LayerSequence : ILayerSequence
    {
        /// <summary>
        ///     The list of layers of the LayerSequence.
        /// </summary>
        protected IList<ILayer> _layers = new List<ILayer>();

        /// <summary>
        ///     Adds a layer to the LayerSequence.
        /// </summary>
        /// 
        /// <param name="layer">
        ///     The layer to add.
        /// </param>
        public void Add(ILayer layer) => _layers.Add(layer);
    }
}
