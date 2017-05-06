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

        private void Initialize ()
        {
            // As an example, we will try to learn a decision machine 
            // that can replicate the "exclusive-or" logical function:
            // Now, we can create the sequential minimal optimization teacher
            var learn = new SequentialMinimalOptimization<Gaussian>()
            {
                UseComplexityHeuristic = true,
                UseKernelEstimation = true,
                CacheSize = regressionData.svmInput.Length / 20
            };

            // And then we can obtain a trained SVM by calling its Learn method
            SupportVectorMachine<Gaussian> svm = learn.Learn(regressionData.svmInput, regressionData.svmOutput);

            // Finally, we can obtain the decisions predicted by the machine:
            double[][] scores = svm.Scores(regressionData.svmInput);
            for(int i=0; i < scores.GetLength(0); i++)
            {
                Debug.WriteLine(i);
                for (int j=0; j < scores[i].Length; i++)
                {
                    Debug.WriteLine("SCORE: " +scores[j]);
                }
            }
        }
    }
}
