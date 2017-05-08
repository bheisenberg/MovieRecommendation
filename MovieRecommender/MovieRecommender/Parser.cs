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
        public string moviesFile = @"Resources\movies.csv";
        public Dictionary<int, double[]> genreDict;
        public List<int> movieIds;
        public List<int> userIds;
        public Parser ()
        {
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
            genreDict = CreateGenreDict();
            //Dictionary<int, int> genreDict = CreateGenreDict();
            return CreateVectors(ratings);
        }

        public Dictionary<int, double[]> CreateGenreDict()
        {
           // Dictionary<int, double[]> genreDict = new Dictionary<int, double[]>();
            List<string> genres = File.ReadLines(moviesFile)
            .Select(csvLine => csvLine.Split(',')).Skip(1)
            .Select(s => s[2].Split('|')).SelectMany(a => a).Distinct().ToList();

            return File.ReadLines(moviesFile)
                .Select(csvLine => csvLine.Split(',')).Skip(1)
                .ToDictionary(s => int.Parse(s[0]), s => CreateVector(genres, s[2].Split('|').ToList()));
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
            int[] svmOutput = new int[ratings.Count];
            for (int i = 0; i < ratings.Count; i++)
            {
                //double[] userVector = CreateVector(userIds, ratings[i].userId);
                //double[] movieVector = CreateVector(movieIds, ratings[i].movieId);
                double[] outputVector = new double[] { ratings[i].rating };
                //input[i] = userVector.Concat(movieVector).ToArray();
                //output[i] = outputVector;
                double[] userMovie = new double[] { ratings[i].userId, ratings[i].movieId };
                input[i] = userMovie.Concat(genreDict[ratings[i].movieId]).ToArray();
                svmOutput[i] = (int)ratings[i].rating * 2;
                output[i] = outputVector;


                //Debug.WriteLine("INPUT LENGTH: " + input[i].Length);
                //PrintVector(input[i], "INPUT");
                //PrintVector(output[i], "OUTPUT");
                //PrintVector(svmInput[i], "SVM INPUT");
                //Debug.WriteLine(svmOutput[i]);
            }
            //NeuralData svmData = new NeuralData(svmInput, svmOutput);
            NeuralData neuralData = new NeuralData(input, output);
            SVMData svmData = new SVMData(input, svmOutput);
            //Debug.WriteLine("FINISHED CREATING VECTORS");
            //Debug.WriteLine("INPUT HEIGHT: " + input.GetLength(0));
            //Debug.WriteLine("INPUT WIDTH: " + input[0].Length);
            //NeuralData regressionData = new NeuralData(svmInput, output);
            return new List<NeuralData>() { neuralData, svmData };
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

        private double[] CreateVector(List<string> classifier, List<string> data)
        {
            double[] vector = new double[classifier.Count];
            for (int i = 0; i < classifier.Count; i++)
            {
                if (data.Contains(classifier[i]))
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
