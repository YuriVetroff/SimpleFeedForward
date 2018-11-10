using SimpleFeedForward.Layers;
using System.Collections.Generic;

namespace SimpleFeedForward
{
    public interface ILayerSequence
    {
        void Add(ILayer layer);
    }

    public abstract class LayerSequence : ILayerSequence
    {
        protected IList<ILayer> _layers = new List<ILayer>();

        public void Add(ILayer layer) => _layers.Add(layer);
    }
}
