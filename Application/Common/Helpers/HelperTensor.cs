using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Numerics.Tensors;

namespace Common.Helpers
{
    public static class HelperTensor
    {
        public static Tensor<float> ConvertImgeToTensor(Bitmap image)
        {
            Tensor<float> result = new DenseTensor<float>(new[] { image.Width, image.Height, 3 });
            Color scanColor; 

            for (int x = 0; x < image.Width; x++)
            {
                for (int y = 0; y < image.Height; y++)
                {
                    scanColor = image.GetPixel(x, y);

                    result[x, y, 0] = scanColor.B / (float)255; 
                    result[x, y, 1] = scanColor.G / (float)255;
                    result[x, y, 2] = scanColor.R / (float)255;  
                }
            }

            return result; 
        }
    }
}
