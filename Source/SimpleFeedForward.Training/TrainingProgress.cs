namespace SimpleFeedForward.Training
{
    public class TrainingProgress
    {
        public int MinibatchIndex { get; set; }
        public int Epoch { get; set; }
        public double? Loss { get; set; }
    }
}
