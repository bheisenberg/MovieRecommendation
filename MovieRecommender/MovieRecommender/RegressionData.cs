using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MovieRecommender
{
    public class RegressionData : NeuralData
    {
        public double[][] svmInput;
        public double[] svmOutput;
        public RegressionData(double[][] input, double[][] output, double[][] svmInput, double[] svmOutput)
        {
            this.input = input;
            this.output = output;
            this.svmInput = svmInput;
            this.svmOutput = svmOutput;
        }

        public int[] RandomIndices(double[][] inputArray)
        {
            int height = inputArray.GetLength(0);
            int[] indicies = new int[height];
            for (int i = 0; i < height; i++)
            {
                indicies[i] = i;
            }
            Random random = new Random();
            return indicies.OrderBy(x => random.Next()).ToArray();
        }

        public RegressionData Shuffle()
        {
            int[] neuralIndices = RandomIndices(input);
            int[] svmIndices = RandomIndices(this.svmInput);
            double[][] neuralInput = new double[neuralIndices.Length][];
            double[][] neuralOutput = new double[neuralIndices.Length][];
            double[][] svmInput = new double[svmIndices.Length][];
            double[] svmOutput = new double[svmIndices.Length];
            for (int i = 0; i < this.input.Length; i++)
            {
                neuralInput[i] = this.input[neuralIndices[i]];
                neuralOutput[i] = this.output[neuralIndices[i]];
            }
            for (int i = 0; i < svmInput.Length; i++)
            {
                svmInput[i] = this.svmInput[svmIndices[i]];
                svmOutput[i] = this.svmOutput[svmIndices[i]];
            }
            return new RegressionData(neuralInput, neuralOutput, svmInput, svmOutput);
        }
    }


}
