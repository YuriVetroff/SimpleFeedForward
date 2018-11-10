using SimpleFeedForward.Data;
using System.Collections.Generic;
using System.Linq;

namespace SimpleFeedForward.Training
{
    public class MinibatchSource
    {
        private readonly DataItem[] _items;
        private int _currentIndex = 0;

        public MinibatchSource(IEnumerable<DataItem> items)
        {
            _items = items.ToArray();
        }

        public bool IsSweepEnd { get; private set; } = false;

        public IEnumerable<DataItem> NextMinibatch(int minibatchSize)
        {
            IsSweepEnd = false;
            var itemsToSkip = _currentIndex;
            var itemsToTake = minibatchSize;
            if (_currentIndex + minibatchSize > _items.Length)
            {
                itemsToTake = _items.Length - _currentIndex;
                _currentIndex = 0;
                IsSweepEnd = true;
            }
            else
            {
                _currentIndex += minibatchSize;
            }
            var minibatch = _items.Skip(itemsToSkip).Take(itemsToTake);
            
            return minibatch;
        }
    }
}
