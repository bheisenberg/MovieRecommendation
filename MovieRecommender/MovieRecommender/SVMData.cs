using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MovieRecommender
{
    public class SVMData : NeuralData
    {
        public new int [] output;
        public SVMData (double[][] input, int[] output)
        {
            this.input = input;
            this.output = output;
        }

        public new SVMData Shuffle()
        {
            int[] indices = RandomIndices(this.input);
            double[][] input = new double[indices.Length][];
            int[] output = new int[indices.Length];
            for (int i = 0; i < this.input.Length; i++)
            {
                input[i] = this.input[indices[i]];
                output[i] = this.output[indices[i]];
            }
            return new SVMData(input, output);
        }
    }


}
