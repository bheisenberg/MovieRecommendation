using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MovieRecommender
{
    public class SVMFold : FoldData
    {
        public new int[] testY;
        public new int[] trainY;
        public SVMFold(double[][] input, int[] output, int start, int end)
        {
            this.trainX = new double[input.GetLength(0) - (end - start)][];
            this.trainY = new int[output.GetLength(0) - (end - start)];
            this.testX = new double[end - start][];
            this.testY = new int[end - start];
            //Debug.WriteLine("END-START: " + (end-start));
            //Debug.WriteLine("TRAINX LENGTH: " + trainX.Length);
            int height = input.GetLength(0);
            int j = 0;
            int k = 0;
            for (int i = 0; i < height; i++)
            {
                if (i >= start && i < end)
                {
                    testX[k] = input[i];
                    testY[k] = output[i];
                    k++;
                }
                else
                {
                    trainX[j] = input[i];
                    trainY[j] = output[i];
                    j++;
                }
            }
        }
    }
}