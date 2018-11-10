namespace SimpleFeedForward.Training.Learners
{
    public class SgdLearner : AbstractLearner
    {
        protected override void FinalizeGradient(double[] gradient)
        {
            for (int i = 0; i < gradient.Length; i++)
            {
                gradient[i] *= -LearningRate;
            }
        }
    }
}
