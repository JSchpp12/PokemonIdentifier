using Microsoft.ML.Data;

namespace Models
{
    internal class OnnxOutput
    {
        [VectorType(859)]
        [ColumnName("dense_1")]
        public float[] PredictedLabels { get; set; }
    }
}
