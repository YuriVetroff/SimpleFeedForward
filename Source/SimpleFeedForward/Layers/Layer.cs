using System.Collections.Generic;
using System.Linq;

namespace SimpleFeedForward.Layers
{
    public interface ILayer : IInitializing<ILayer>, IInitializing<int>, IProducing
    {
        #region Properties

        int NeuronCount { get; }
        IReadOnlyList<double> Error { get; }

        #endregion

        #region Methods

        void Forward();
        void Backward();

        void SetInput(double[] input);
        void SetError(double[] error);

        #endregion
    }

    public abstract class Layer : ILayer
    {
        #region Fields

        protected readonly Engine _engine = Engine.Instance;

        protected int _incomingLength;
        protected int _outgoingLength;

        protected double[] _input;
        protected double[] _output;
        protected double[] _error;
        protected double[] _backError;

        protected ILayer _incomingLayer;

        #endregion

        #region Properties

        public int NeuronCount => _outgoingLength;

        public virtual IReadOnlyList<double> Output => _output;
        public IReadOnlyList<double> Error => _error;

        #endregion

        #region Methods

        public virtual void Forward()
        {
            if (_incomingLayer != null)
            {
                SetInput(_incomingLayer.Output.ToArray());
            }
        }
        public virtual void Backward()
        {
            if (_incomingLayer != null)
            {
                _incomingLayer.SetError(_backError);
            }
        }

        public virtual void Init(ILayer incoming)
        {
            _incomingLayer = incoming;
            Init(incoming.NeuronCount);
        }
        public virtual void Init(int incomingLength)
        {
            _incomingLength = incomingLength;
            _input = new double[incomingLength];
            _backError = new double[incomingLength];
        }

        public void SetInput(double[] input) => input.CopyTo(_input, 0);
        public void SetError(double[] error) => error.CopyTo(_error, 0);

        #endregion
    }
}
