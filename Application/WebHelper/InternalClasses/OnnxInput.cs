using Microsoft.ML.Data;
using System.Drawing;
using Microsoft.ML;
using Microsoft.ML.Transforms.Image;

namespace Models
{
    internal class OnnxInput
    {
        //[VectorType(80 * 80 * 1)]
        //[ColumnName("conv2d_input")]
        //public float[] ImageData { get; set; }

        [ColumnName("bitmap")]
        [ImageType(80, 80)]
        public Bitmap ImageData { get; set; }


    }
}
