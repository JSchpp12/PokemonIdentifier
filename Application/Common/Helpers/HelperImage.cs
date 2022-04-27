using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing; 

namespace Common.Helpers
{
    public static class HelperImage
    {
        
        //public static float[][] NormalizeImage(Bitmap image)
        //{
            //float normR, normG, normB; //normColors
            //var result = new float[image.Width][]; 

            //for (int i =0; i < image.Width; i++)
            //{
            //    result[i] = new float[image.Height]; 

            //    for (int j = 0; j < image.Height; j++)
            //    {
            //        var currentPixel = image.GetPixel(i, j);
            //        normR = currentPixel.R / (float)255;
            //        normG = currentPixel.G / 255; 
            //        normB = currentPixel.B / 255;

            //        resultBitmap.SetPixel(i, j, Color.FromArgb(normR, normG, normB));
            //    }
            //}

            //return resultBitmap; 

            //int xStride = image.Height * 3;
            //int yStride = 3;
            //Color scanColor;

            //float[] data = new float[image.Width * image.Height * 3];
            //for (int x = 0; x < image.Width; x++)
            //{
            //    for (int y = 0; y < image.Height; y++)
            //    {
            //        scanColor = image.GetPixel(x, y);
            //        data[x * xStride + y * yStride + 0] = scanColor.B;
            //        data[x * xStride + y * yStride + 1] = scanColor.G;
            //        data[x * xStride + y * yStride + 2] = scanColor.R;
            //    }
            //}

            //return data;
        //}
    }
}
