using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

public class FoldData
{
    public virtual double[][] trainX { get; set; }
    public virtual double[][] trainY { get; set; }
    public virtual double[][] testX { get; set; }
    public virtual double[][] testY { get; set; }
    //public int[] trainYint { get; set; }
    //public int[] testYint { get; set; }

    public FoldData(double[][] input, double[][] output, int start, int end)
    {
        trainX = new double[input.GetLength(0) - (end - start)][];
        trainY = new double[output.GetLength(0) - (end - start)][];
        testX = new double[end - start][];
        testY = new double[end - start][];
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

    public FoldData ()
    {

    }


    /*public FoldData(double[][] input, int[] output, int start, int end)
    {
        this.trainX = new double[input.GetLength(0) - (end - start)][];
        this.trainYint = new int[output.GetLength(0) - (end - start)];
        this.testX = new double[end - start][];
        this.testYint = new int[end - start];
        Debug.WriteLine("END-START: " + (end - start));
        Debug.WriteLine("TRAINX LENGTH: " + trainX.Length);
        int height = input.GetLength(0);
        int j = 0;
        int k = 0;
        for (int i = 0; i < height; i++)
        {
            if (i >= start && i < end)
            {
                testX[k] = input[i];
                testYint[k] = output[i];
                k++;
            }
            else
            {
                trainX[j] = input[i];
                trainYint[j] = output[i];
                j++;
            }
        }
    }*/
}
