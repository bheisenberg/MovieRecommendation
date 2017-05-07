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
        private RegressionData regressionData;
        public MovieRecommender(RegressionData regressionData)
        {
            this.regressionData = regressionData.Shuffle();
            Initialize();
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
            int n = regressionData.svmInput.GetLength(0);
            int size = n / folds;
            for (int i = 0; i < folds; i++)
            {
                Debug.WriteLine("STARTING FOLD " + i);
                int start = i * size;
                int end = (i + 1) * size - 1;
                FoldData currentFold = new FoldData(regressionData.svmInput, regressionData.svmOutput, start, end);
                // Use Ordinary Least Squares to learn the regression
                OrdinaryLeastSquares ols = new OrdinaryLeastSquares();
                // Use OLS to learn the simple linear regression
                var regression = ols.Learn(currentFold.trainX, currentFold.svmTrainY);

                // Compute the output for a given input:
                for (int j = 0; j < currentFold.testX.GetLength(0); j++)
                {
                    double y = regression.Transform(currentFold.testX[j]); // The answer will be 28.088
                    Debug.WriteLine(y);
                }
            }
        }
    }
}
