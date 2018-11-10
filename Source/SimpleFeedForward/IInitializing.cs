namespace SimpleFeedForward
{
    public interface _IInitializing { }

    public interface IInitializing : _IInitializing
    {
        void Init();
    }

    public interface IInitializing<T> : _IInitializing
    {
        void Init(T parameter);
    }
}