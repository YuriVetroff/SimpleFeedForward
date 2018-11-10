namespace SimpleFeedForward.Layers
{
    public class FullyConnectedLayer : BiasLayer
    {
        public FullyConnectedLayer(int neuronCount)
        {
            _outgoingLength = neuronCount;

            _output = new double[neuronCount];
            _error = new double[neuronCount];
        }
    }
}
