using SimpleFeedForward.Layers;
using System.Collections.Generic;
using System.Linq;

namespace SimpleFeedForward
{
    public interface INetwork : ITrainable, IProducing, IInitializing, ILayerSequence
    {
        void Forward(double[] input);
        void Backward(double[] error);
    }

    public class Network : LayerSequence, INetwork
    {
        private InputLayer _inputLayer;

        public Network(int inputSize)
        {
            _inputLayer = new InputLayer(inputSize);
        }

        public IReadOnlyList<double> Output => _layers.Last().Output;

        public void Forward(double[] input)
        {
            _inputLayer.SetInput(input);
            _inputLayer.Forward();

            foreach (var layer in _layers)
            {
                layer.Forward();
            }
        }
        public void Backward(double[] error)
        {
            _layers.Last().SetError(error);
            foreach (var layer in _layers.Reverse())
            {
                layer.Backward();
            }
        }

        public void Init()
        {
            _layers[0].Init(_inputLayer);
            for (int i = 1; i < _layers.Count; i++)
            {
                _layers[i].Init(_layers[i - 1]);
            }
        }

        public IEnumerable<DataFlow> ExtractData()
        {
            var data = new List<DataFlow>();

            foreach (ITrainable weightLayer in _layers.Where(layer => layer is ITrainable))
            {
                data.AddRange(weightLayer.ExtractData());
            }

            return data;
        }
    }
}
