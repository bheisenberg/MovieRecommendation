using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;

namespace MovieRecommender
{
    public class Parser
    {
        public string ratingsFile = @"Resources\ratings.csv";
        public Dictionary<int, List<MovieRating>> userData;
        public List<int> movieIds;
        public List<int> userIds;
        public Parser ()
        {
            userData = new Dictionary<int, List<MovieRating>>();
            movieIds = new List<int>();
            userIds = new List<int>();
            GetNeuralData();
        }

        public List<NeuralData> GetNeuralData ()
        {
            List<MovieRating> ratings = File.ReadLines(ratingsFile)
                .Select(csvLine => csvLine.Split(',')).Skip(1)
                .Select(s => new MovieRating(int.Parse(s[0]), int.Parse(s[1]), double.Parse(s[2]), int.Parse(s[3]))).ToList();
            GetUserData(ratings);
            return CreateVectors(ratings);
        }

        private void GetUserData (List<MovieRating> ratings)
        {
            foreach (MovieRating rating in ratings)
            {
                if (!movieIds.Contains(rating.movieId)) movieIds.Add(rating.movieId);
                if (!userIds.Contains(rating.userId)) userIds.Add(rating.userId);
            }
        }

        private List<NeuralData> CreateVectors(List<MovieRating> ratings)
        {
            double[][] input = new double[ratings.Count][];
            double[][] output = new double[ratings.Count][];
            //double[][] svmInput = new double[ratings.Count][];
            //double[] svmOutput = new double[ratings.Count];
            for (int i = 0; i < 1000; i++)
            {
                double[] userVector = CreateVector(userIds, ratings[i].userId);
                double[] movieVector = CreateVector(movieIds, ratings[i].movieId);
                double[] outputVector = new double[] { ratings[i].rating };
                input[i] = userVector.Concat(movieVector).ToArray();
                output[i] = outputVector;
                //svmInput[i] = new double[] { ratings[i].userId, ratings[i].movieId };
                //svmOutput[i] = ratings[i].rating;
                //PrintVector(input[i], "INPUT");
                //PrintVector(output[i], "OUTPUT");
                //PrintVector(svmInput[i], "SVM INPUT");
                //Debug.WriteLine(svmOutput[i]);
            }
            NeuralData neuralData = new NeuralData(input, output);
            Debug.WriteLine("FINISHED CREATING VECTORS");
            //NeuralData regressionData = new NeuralData(svmInput, output);
            return new List<NeuralData>() { neuralData };
        }



        private double[] CreateVector(List<int> classifier, int data)
        {
            double[] vector = new double[classifier.Count];
            for (int i = 0; i < classifier.Count; i++)
            {
                if (data == classifier[i])
                {
                    vector[i] = 1;
                }
                else
                {
                    vector[i] = 0;
                }
            }
            return vector;
        }

        private void PrintVector(double[] input, string name)
        {
            string inputPrint = name+": < ";
            for (int j = 0; j < input.Length; j++)
            {
                inputPrint += input[j] + ", ";
            }
            inputPrint += ">";
            Debug.WriteLine(inputPrint);
        }
    }
}
