using System.Collections.Generic;

namespace SimpleFeedForward
{
    public interface IProducing
    {
        IReadOnlyList<double> Output { get; }
    }
}
