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
        public string testFile = @"Resources\test.csv"; //File containing data for a trained neural network to predict on
        public string trainFile = @"Resources\ratings.csv"; //File containing data for an untrained neural net to train on
        public string genreFile = @"Resources\movies.csv"; //File containing genre data
        public string activeFile;
        public Dictionary<int, double[]> genreDict;
        public List<int> movieIds;
        public List<int> userIds;
        public Parser ()
        {
            movieIds = new List<int>();
            userIds = new List<int>();
            activeFile = testFile; //testFile for testing, ratingsFile for training
        }

        public ReccomenderData GetNeuralData ()
        {
            //Parses the active file (test or train) into a list of ratings, ready for vectorization
            List<MovieRating> ratings = File.ReadLines(activeFile)
                .Select(csvLine => csvLine.Split(','))
                .Select(s => new MovieRating(int.Parse(s[0]), int.Parse(s[1]), 0, 0)).ToList();

            //Parses the train file to create the basis for the vectors
            List<MovieRating> vectorData = File.ReadLines(trainFile)
                .Select(csvLine => csvLine.Split(',')).Skip(1)
                .Select(s => new MovieRating(int.Parse(s[0]), int.Parse(s[1]), double.Parse(s[2]), 0)).ToList();
            GetUserData(vectorData);
            genreDict = CreateGenreDict();

            return CreateVectors(ratings);
        }

        public Dictionary<int, double[]> CreateGenreDict()
        {
            //Reads in the genre index of the genres file, splits the topics and adds distinct ones to a list
            List<string> genres = File.ReadLines(genreFile)
            .Select(csvLine => csvLine.Split(',')).Skip(1)
            .Select(s => s[2].Split('|')).SelectMany(a => a).Distinct().ToList();

            //Creates a dictionary entry with movieId as a key and a vector of genres as a value
            return File.ReadLines(genreFile)
                .Select(csvLine => csvLine.Split(',')).Skip(1)
                .ToDictionary(s => int.Parse(s[0]), s => CreateVector(genres, s[2].Split('|').ToList()));
        }

        //Compiles a list of unique user data
        private void GetUserData (List<MovieRating> ratings)
        {
            foreach (MovieRating rating in ratings)
            {
                if (!movieIds.Contains(rating.movieId)) movieIds.Add(rating.movieId);
                if (!userIds.Contains(rating.userId)) userIds.Add(rating.userId);
            }
        }

        private ReccomenderData CreateVectors(List<MovieRating> ratings)
        {
            Debug.WriteLine("CREATING VECTORS");
            double[][] input = new double[ratings.Count][];
            double[][] output = new double[ratings.Count][];
            double[][] svmInput = new double[ratings.Count][];
            double movieMax = movieIds.Max();
            double userMax = userIds.Max();
            int[] svmOutput = new int[ratings.Count];
            for (int i = 0; i < ratings.Count; i++)
            {
                double[] userVector = CreateVector(userIds, ratings[i].userId);
                double[] outputVector = new double[] { ratings[i].rating / 5 };
                double[] movieVector = new double[] { ratings[i].movieId / movieMax };
                input[i] = userVector.Concat(movieVector).Concat(genreDict[ratings[i].movieId]).ToArray();
                output[i] = outputVector;
                svmInput[i] = new double[] { ratings[i].userId, ratings[i].movieId, }.Concat(genreDict[ratings[i].movieId]).ToArray();
                svmOutput[i] = (int)ratings[i].rating;
            }
            NeuralData neuralData = new NeuralData(input, output, ratings);
            SVMData svmData = new SVMData(input, svmOutput);
            Debug.WriteLine("FINISHED CREATING VECTORS");
            if (Program.verbose == 1)
            {
                Debug.WriteLine("INPUT HEIGHT: " + input.GetLength(0));
                Debug.WriteLine("INPUT WIDTH: " + input[0].Length);
            }
            NeuralData regressionData = new NeuralData(svmInput, output);
            return new ReccomenderData(neuralData, svmData);
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
