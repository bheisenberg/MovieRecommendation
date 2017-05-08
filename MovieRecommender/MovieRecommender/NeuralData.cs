using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

public class NeuralData
{
    public double[][] input { get; set; }
    public double[][] output { get; set; }
    public int[] intOutput { get; set; }
    public string[] topics { get; set; }

    public NeuralData(double[][] input, double[][] output, string[] topics)
    {
        this.input = input;
        this.output = output;
        this.topics = topics;
    }

    public NeuralData(double[][] input, double[][] output)
    {
        this.input = input;
        this.output = output;
    }

    public NeuralData(double[][] input, int[] output)
    {
        this.input = input;
        this.intOutput = output;
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

    public NeuralData Shuffle()
    {
        int[] neuralIndices = RandomIndices(input);
        double[][] neuralInput = new double[neuralIndices.Length][];
        double[][] neuralOutput = new double[neuralIndices.Length][];
        for (int i = 0; i < this.input.Length; i++)
        {
            neuralInput[i] = this.input[neuralIndices[i]];
            neuralOutput[i] = this.output[neuralIndices[i]];
        }
        return new NeuralData(neuralInput, neuralOutput);
    }

    public override string ToString()
    {
        string stringOutput = "";
        for (int i = 0; i < input.Length; i++)
        {
            stringOutput += "input: < ";
            for (int j = 0; j < input[i].Length; j++)
            {
                stringOutput += (string.Format("{0}, ", input[i][j]));
            }
            stringOutput += (">\n");
        }
        return stringOutput;
    }
}

