//
// Yuri Vetroff
// yuri.vetroff@gmail.com
//

namespace SimpleFeedForward
{
    /// <remarks>
    ///     Defined for future use.
    /// </remarks>
    public interface _IInitializing { }

    /// <summary>
    ///     Defines a model that requires initialization.
    /// </summary>
    public interface IInitializing : _IInitializing
    {
        /// <summary>
        ///     Initializes the IInitializing.
        /// </summary>
        void Init();
    }

    /// <summary>
    ///     Defines a model that requires initialization
    ///     with a parameter of some type.
    /// </summary>
    /// 
    /// <typeparam name="T">
    ///     The type of the initialization parameter.
    /// </typeparam>
    public interface IInitializing<T> : _IInitializing
    {
        /// <summary>
        ///     Initializes the IInitializing with the specified parameter.
        /// </summary>
        /// 
        /// <param name="parameter">
        ///     The initializing parameter.
        /// </param>
        void Init(T parameter);
    }
}