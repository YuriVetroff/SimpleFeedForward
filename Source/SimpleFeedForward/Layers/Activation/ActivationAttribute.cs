//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

using System;

namespace SimpleFeedForward.Layers
{
    public class ActivationAttribute : Attribute
    {
        public ActivationAttribute(ActivationType activationType)
        {
            ActivationType = activationType;
        }

        public ActivationType ActivationType { get; private set; }
    }
}
