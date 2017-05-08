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
using Accord.Neuro.Learning;
using Accord.MachineLearning.VectorMachines;

namespace MovieRecommender
{
    class MovieRecommender
    {
        private int folds = 1;
        private float targetError = 0.001f;
        private List<NeuralData> data;
        public MovieRecommender(List<NeuralData> data)
        {
            ShuffleData(data);
            NeuralNetwork();
            //SVM();
        }

        private void ShuffleData(List<NeuralData> data)
        {
            this.data = new List<NeuralData>();
            foreach(NeuralData item in data)
            {
                this.data.Add(item);
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

        private void NeuralNetwork ()
        {
            int n = data[0].input.GetLength(0);
            int size = n / folds;
            for (int i = 0; i < folds; i++)
            {
                Debug.WriteLine("CURRENT FOLD: " + i);
                int start = i * size;
                int end = (i + 1) * size - 1;
                FoldData currentFold = new FoldData(data[0].input, data[0].output, start, end);
                int hiddenLayers = (currentFold.trainX[0].Length + currentFold.trainY[0].Length) / 2;
                Debug.WriteLine(hiddenLayers);
                IActivationFunction function = new BipolarSigmoidFunction();
                var network = new ActivationNetwork(function, currentFold.trainX[0].Length, hiddenLayers, currentFold.trainY[0].Length);
                var teacher = new BackPropagationLearning(network);
                teacher.Momentum = 0.9f;
                teacher.LearningRate = 0.1f;
                new NguyenWidrow(network).Randomize();
                double error = int.MaxValue;
                double previousError = error;
                Debug.WriteLine("INITIAL ERROR: " + error);
                do
                {
                    previousError = error;
                    error = teacher.RunEpoch(currentFold.trainX, currentFold.trainY);
                    Debug.WriteLine("ERROR: " + error + " CHANGE: " + (Math.Abs(previousError - error) / previousError));
                }
                while (error > 0 && (Math.Abs(previousError - error)) / previousError > targetError);
                double[][] predictedY = ComputeOutput(network, currentFold.testX);
                /*for(int j=0; j < predictedY.GetLength(0); j++)
                {
                    for(int k=0; k < predictedY[j].Length; j++)
                    {
                        Debug.WriteLine(predictedY[j][k]);
                    }
                }*/
            }
        }

        public double[][] ComputeOutput(ActivationNetwork network, double[][] testX)
        {
            double[][] tempOutput = new double[testX.GetLength(0)][];
            for (int i = 0; i < testX.GetLength(0); i++)
            {
                tempOutput[i] = network.Compute(testX[i]);
            }
            return tempOutput;
        }

        /*private void SVM ()
        {
            int n = data[0].input.GetLength(0);
            int size = n / folds;
            for (int i = 0; i < folds; i++)
            {
                Debug.WriteLine("STARTING FOLD " + i);
                int start = i * size;
                int end = (i + 1) * size - 1;
                FoldData currentFold = new FoldData(data[0].input, data[0].intOutput, start, end);
                var teacher = new MulticlassSupportVectorLearning<Gaussian>()
                {
                    // Configure the learning algorithm to use SMO to train the
                    //  underlying SVMs in each of the binary class subproblems.
                    Learner = (param) => new SequentialMinimalOptimization<Gaussian>()
                    {
                        // Estimate a suitable guess for the Gaussian kernel's parameters.
                        // This estimate can serve as a starting point for a grid search.
                        UseKernelEstimation = true
                    }
                };

                // Learn a machine
                var machine = teacher.Learn(data[0].input, data[0].intOutput);


                // Create the multi-class learning algorithm for the machine
                var calibration = new MulticlassSupportVectorLearning<Gaussian>()
                {
                    Model = machine, // We will start with an existing machine

                    // Configure the learning algorithm to use SMO to train the
                    //  underlying SVMs in each of the binary class subproblems.
                    Learner = (param) => new ProbabilisticOutputCalibration<Gaussian>()
                    {
                        Model = param.Model // Start with an existing machine
                    }
                };


                // Configure parallel execution options
                calibration.ParallelOptions.MaxDegreeOfParallelism = 1;

                // Learn a machine
                calibration.Learn(data[0].input, data[0].intOutput);

                // Obtain class predictions for each sample
                int[] predicted = machine.Decide(data[0].input);

                for(int j=0; j < predicted.Length; j++)
                {
                    Debug.WriteLine(predicted[i]);
                }

                // Get class scores for each sample
                //double[] scores = machine.Score(inputs);
            }
        }*/

        private void LeastSquares()
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
