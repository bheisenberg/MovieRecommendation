using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using Accord.Statistics.Models.Regression;
using Accord.Statistics.Models.Regression.Linear;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;
using Accord;
using Accord.Neuro;
using Accord.MachineLearning.VectorMachines;

namespace MovieRecommender
{
    class MovieRecommender
    {
        private int folds = 10;
        private List<NeuralData> data;
        public MovieRecommender(List<NeuralData> data)
        {
            ShuffleData(data);
            Initialize();
        }

        private void ShuffleData(List<NeuralData> data)
        {
            this.data = new List<NeuralData>();
            foreach(NeuralData item in data)
            {
                this.data.Add(item.Shuffle());
            }
        }

        private void PrintVector(double[] input, string name)
        {
            string inputPrint = name + ": < ";
            for (int j = 0; j < input.Length; j++)
            {
                inputPrint += input[j] + ", ";
            }
            inputPrint += ">";
            Debug.WriteLine(inputPrint);
        }

        private void Initialize()
        {
            int n = data[1].input.GetLength(0);
            int size = n / folds;
            for (int i = 0; i < folds; i++)
            {
                Debug.WriteLine("STARTING FOLD " + i);
                int start = i * size;
                int end = (i + 1) * size - 1;
                FoldData currentFold = new FoldData(data[1].input, data[1].output, start, end);
                // Use Ordinary Least Squares to learn the regression
                OrdinaryLeastSquares ols = new OrdinaryLeastSquares();
                // Use OLS to learn the simple linear regression
                var regression = ols.Learn(currentFold.trainX, currentFold.trainY);

                // Compute the output for a given input:
                double [][] testOutput = regression.Transform(currentFold.testX); // The answer will be 28.088

                for(int j=0; j < testOutput.GetLength(0); j++)
                {
                    for(int k=0; k < testOutput[j].Length; i++)
                    {
                        Debug.WriteLine(testOutput[k]);
                    }
                }
            }
        }
    }
}
