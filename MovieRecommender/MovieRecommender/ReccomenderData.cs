using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MovieRecommender
{
    public class ReccomenderData
    {
        public NeuralData neuralData;
        public SVMData svmData;

        public ReccomenderData(NeuralData neuralData, SVMData svmData)
        {
            this.neuralData = neuralData;
            this.svmData = svmData;
        }
    }
}
