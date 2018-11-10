namespace SimpleFeedForward.Data
{
    public class DataItem
    {
        public double[] Input { get; set; }
        public double[] Output { get; set; }

        public static DataItem Create(int inputSize, int outputSize)
            => new DataItem
            {
                Input = new double[inputSize],
                Output = new double[outputSize]
            };
    }
}
