using System.Collections.Generic;

namespace SimpleFeedForward
{
    public interface ITrainable
    {
        IEnumerable<DataFlow> ExtractData();
    }
}
