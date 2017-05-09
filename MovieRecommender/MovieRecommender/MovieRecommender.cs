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
using Accord.Neuro.Networks;
using Accord.MachineLearning.VectorMachines;
using System.IO;

namespace MovieRecommender
{
    class MovieRecommender
    {
        private int folds = 10;
        private bool train = false;
        private float targetError = 0.001f;
        private ReccomenderData data;
        public string networkFile = @"Resources\network.txt";
        public MovieRecommender(ReccomenderData data)
        {
            this.data = data;
            if (train) NeuralNetwork(); else LoadNeuralNetwork();
        }

        private void LoadNeuralNetwork()
        {
            Debug.WriteLine("LOADING NETWORK FROM FILE");
            var currData = data.neuralData;
            var network = DeepBeliefNetwork.Load(networkFile);
            var teacher = new BackPropagationLearning(network);
            double[][] predictedY = ComputeOutput(network, currData.input);
            double yBar = predictedY.Select(s => s.Sum()).Sum() / predictedY.GetLength(0);
            double rSquared = calculateRSquared(predictedY, currData.output);
            Debug.WriteLine("R SQUARED: " + rSquared);
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
            var currData = data.neuralData.Shuffle();
            double totalRSquared = 0;
            Debug.WriteLine("STARTING NEURAL NETWORK");
            int n = currData.input.Take(1000).ToArray().GetLength(0);
            int size = n / folds;
            for (int i = 0; i < folds; i++)
            {
                Debug.WriteLine("CURRENT FOLD: " + i);
                int start = i * size;
                int end = (i + 1) * size - 1;
                FoldData currentFold = new FoldData(currData.input.Take(1000).ToArray(), currData.output.Take(1000).ToArray(), start, end);
                int hiddenLayers = (currentFold.trainX[0].Length + currentFold.trainY[0].Length) / 2;
                Debug.WriteLine(hiddenLayers);
                IActivationFunction function = new BipolarSigmoidFunction();
                var network = new DeepBeliefNetwork(currentFold.trainX[0].Length, hiddenLayers, currentFold.trainY[0].Length);
                network.SetActivationFunction(function);
                var teacher = new BackPropagationLearning(network);
                teacher.Momentum = 0.9f;
                teacher.LearningRate = 0.3f;
                new NguyenWidrow(network).Randomize();
                double error = int.MaxValue;
                double previousError = error;
                double delta;
                Debug.WriteLine("INITIAL ERROR: " + error);
                do
                {
                    previousError = error;
                    error = teacher.RunEpoch(currentFold.trainX, currentFold.trainY);
                    delta = (Math.Abs(previousError - error) / previousError);
                    Debug.WriteLine("ERROR: " + error + " CHANGE: " + delta);
                }
                while (error > 0 && delta > targetError);
                double[][] predictedY = ComputeOutput(network, currentFold.testX);
                double rSquared = calculateRSquared(predictedY, currentFold.testY);
                totalRSquared += rSquared;
                Debug.WriteLine("R SQUARED: " +rSquared);
                if (i == 0)
                {
                    Debug.WriteLine("SAVED");
                    File.Delete(networkFile);
                    File.Create(networkFile).Close();
                    network.Save(networkFile);
                }
            }

            Debug.WriteLine("AVERAGE R SQUARED: " + totalRSquared / folds);
        }

        public double calculateRSquared(double[][] predictedY, double[][] actualY)
        {
            double RSS = 0;
            double TSS = 0;
            double yBar = predictedY.Select(s => s.Sum()).Sum() / predictedY.GetLength(0);

            /*double rss = (from predicted in predictedY
                          from actual in actualY
                          from predictedValue in predicted
                          from actualValue in actual
                          select Math.Pow(predictedValue - actualValue, 2)).Sum();

            Debug.WriteLine("RSS: " + rss);

            double tss = (from actual in actualY
                          from actualValue in actual
                          select Math.Pow(actualValue - yBar, 2)).Sum();*/
            for (int j = 0; j < predictedY.GetLength(0); j++)
            {
                for (int k = 0; k < predictedY[j].Length; k++)
                {
                    double predicted = predictedY[j][k];
                    double actual = actualY[j][k];
                    Debug.WriteLine("Predicted: " + predicted + " Actual: " + actual);
                    double rssIncrementer = Math.Pow(predicted - actual, 2);
                    double tssIncrementer = Math.Pow(actual - yBar, 2);
                    RSS += rssIncrementer;
                    TSS += tssIncrementer;
                }
            }
            Debug.WriteLine("RSS ITERATIVE: " + RSS);
            Debug.WriteLine("TSS ITERATIVE: " + TSS);
            return 1 - (RSS / TSS);
        }

        public double[][] ComputeOutput(DeepBeliefNetwork network, double[][] testX)
        {
            double[][] tempOutput = new double[testX.GetLength(0)][];
            for (int i = 0; i < testX.GetLength(0); i++)
            {
                tempOutput[i] = network.Compute(testX[i]);
            }
            return tempOutput;
        }

        private void SVM()
        {
            SVMData currData = data.svmData.Shuffle();
            int n = currData.input.GetLength(0);
            int size = n / folds;
            for (int i = 0; i < folds; i++)
            {
                if (i > 0) break;
                Debug.WriteLine("STARTING FOLD " + i);
                Debug.WriteLine(currData.output.Length);
                int start = i * size;
                int end = (i + 1) * size - 1;
                SVMFold currentFold = new SVMFold(currData.input, currData.output, start, end);
                //currentFold.trainX = currentFold.trainX.Concat(currentFold.trainX).ToArray();
                //currentFold.trainY = currentFold.trainY.Concat(currentFold.trainY).ToArray();

                currentFold.trainX = currentFold.trainX.Take(10000).ToArray();
                currentFold.trainY = currentFold.trainY.Take(10000).ToArray();
                Debug.WriteLine("INPUT VECTOR SIZE: " + currentFold.trainX[0].Length);
                Debug.WriteLine("INPUT SIZE: " + currentFold.trainX.GetLength(0));
                Debug.WriteLine("OUTPUT SIZE: " + currentFold.trainY.Length);

                // currentFold.trainX = inputs.Take(4).ToArray<double[]>();
                // currentFold.trainY = outputs.Take(4).ToArray<int>();
                // currentFold.trainX = currentFold.trainX.Take(10).ToArray<double[]>();
                /*for (int j = 0; j < currentFold.trainY.Length; j++)
                    currentFold.trainY[j] = currentFold.trainY[j] - 1;*/
                // currentFold.trainX = new double[][] { new double[] { 0, 0, 0, 0 }, new double[] { 1, 1, 1, 1 } };
                // currentFold.trainY = new int[] {  0, 1, 0, 1, 0, 0, 0, 1, 1, 0 };
                /*for (int j = 0; j < currentFold.trainX.Length; j++)
                {
                    PrintVector(currentFold.trainX[j], "INPUT");
                    if (currentFold.trainX[j].Length != currentFold.trainX[0].Length) flag = true;
                    Debug.WriteLine("OUTPUT " + currentFold.trainY[j]);
                }*/

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
                try
                {
                    var machine = teacher.Learn(currentFold.trainX, currentFold.trainY);
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
                    calibration.Learn(currentFold.trainX, currentFold.trainY);

                    // Obtain class predictions for each sample
                    int[] predicted = machine.Decide(currentFold.testX);

                    for (int j = 0; j < predicted.Length; j++)
                    {
                        Debug.WriteLine(predicted[j] + " " + currentFold.testY[j]);
                    }
                }
                catch (AggregateException ex)
                {
                    Debug.WriteLine(ex.InnerException.Message);
                }

                // Get class scores for each sample
                //double[] scores = machine.Score(inputs);
            }
        }



        /*private void LeastSquares()
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
         }*/
    }
}
