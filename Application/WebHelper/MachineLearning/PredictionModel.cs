using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Linq;
using System.Threading.Tasks;
using System.Drawing; 
using Microsoft.ML.OnnxRuntime;
using Common.Helpers;
using Microsoft.ML;
using NumSharp; 
using System.Reflection;

namespace Models.MachineLearning
{
    public class PredictionModel
    {
        static readonly string modelPath = "./Models/identifier_v1.onnx";
        //private static MLContext mlContext;
        //private static ITransformer predictionPipeline;

        //private static PredictionEngine<OnnxInput, OnnxOutput> predictionEngine;
        //private static ITransformer trainedModel; 

        /// <summary>
        /// Initalize the needed MLContext and other needed datasets for predictions. 
        /// </summary>
        /// <exception cref="FileNotFoundException">Thrown if the model file cannot be loaded.</exception>
        /// <exception cref="Exception">Thrown if some unknown error occurrs while loading the data.</exception>
        static PredictionModel()
        {
            //prepare model 
            //mlContext = new MLContext();

            //if (!File.Exists(modelPath))
            //{
            //    throw new FileNotFoundException(modelPath);
            //}

            //PredictionModel.predictionPipeline = GetPredictionPipeline(); 

            //using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            //{
            //    trainedModel = mlContext.Model.Load(stream, out _);
            //}

            //if (trainedModel == null)
            //{
            //    throw new Exception($"Failed to load Model");
            //}


        }

        /// <summary>
        /// Run a prediction on the model from the given image.
        /// </summary>
        /// <param name="predictImage">Image to run model prediction on</param>
        /// <returns>The prediction information</returns>
        public static void ClassifyImage(Bitmap imageData)
        {
            //normalize image
            //Bitmap normImage = HelperImage.NormalizeImage(predictImage);

            ////run prediction

            //using var session = new InferenceSession(modelPath);

            //var modelInputLayerName = session.InputMetadata.Keys.Single();
            //var imageFlattened = normImage.SelectMany(x => y).ToArray();
            //MLContext mlContext = new MLContext();
            //var test = HelperTensor.ConvertImgeToTensor(imageData);
            ////var flat = test.SelectMany(x => x).ToArray(); 

            //var input = new OnnxInput { ImageData = test.ToArray<float>() };
            //var testList = new List<OnnxInput>() { image };

            //var dataView = mlContext.Data.LoadFromEnumerable(testList);
            //var pipeline = mlContext.Transforms.ApplyOnnxModel(outputColumnName: "dense_1", inputColumnName: "conv2d_1", modelPath);
            //var transformedValues = pipeline.Fit(dataView).Transform(dataView);
            //var output = mlContext.Data.CreateEnumerable<OnnxOutput>(transformedValues, reuseRowObject: false);

            //var emptyData = new List<OnnxInput>();
            //var content = np.Load()
            //var data = np.array(content[imageData]);
            //var data = mlContext.Data.LoadFromEnumerable(emptyData);

            //var pipeline = mlContext.Transforms.ResizeImages(resizing: Microsoft.ML.Transforms.Image.ImageResizingEstimator.ResizingKind.Fill,
            //    outputColumnName: "conv2d_input", imageWidth: 80, imageHeight: 80)
            //    .Append(mlContext.Transforms.NormalizeBinning(inputColumnName: "conv2d_input", outputColumnName: "dense_1"))
            //    .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "conv2d_input", outputAsFloatArray: true))
            //    .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: modelPath, outputColumnName: "dense_1", inputColumnName: "conv2d_input"));

            //var model = pipeline.Fit(data);

            //var predictionEngine = mlContext.Model.CreatePredictionEngine<OnnxInput, OnnxOutput>(model);
            //var prediction = predictionEngine.Predict(new OnnxInput { ImageData = imageData});

            //var onnxPredictionEngine = mlContext.Model.CreatePredictionEngine<OnnxInput, OnnxOutput>(predictionPipeline); 
            //var testInput = new OnnxInput { ImageData = }

            var mlContext = new MLContext();
            //var emptyData = new List<OnnxInput>(); 
            var emptyData = new List<OnnxInput>() { new OnnxInput { ImageData = imageData } };  

            var dataView = mlContext.Data.LoadFromEnumerable(emptyData);
            var pipeline =
                mlContext.Transforms.ResizeImages(
                   outputColumnName: "conv2d_input", //will need to change this to an immediate value 
                   imageWidth: 80,
                   imageHeight: 80,
                   inputColumnName: "bitmap",
                   resizing: Microsoft.ML.Transforms.Image.ImageResizingEstimator.ResizingKind.IsoPad) //unsure about inpurColumnName here
                                                                                                       //.Append(mlContext.Transforms.NormalizeBinning(outputColumnName: "", inputColumnName: "bitmap"))
                .Append(mlContext.Transforms.ExtractPixels(
                    outputColumnName: "conv2d_input", interleavePixelColors: false)) //unsure about the interleaved pixel colors
                .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: modelPath, inputColumnName: "conv2d_input", outputColumnName: "dense_1"));

            //var model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<OnnxInput>()));//die here --need to fix input types 
            var model = pipeline.Fit(dataView); 
            var predictionEngine = mlContext.Model.CreatePredictionEngine<OnnxInput, OnnxOutput>(model);
            var inputDatas = new OnnxInput { ImageData = imageData };
            
            var prediction = predictionEngine.Predict(inputDatas);
        }

        //static ITransformer GetPredictionPipeline()
        //{
        //    var onnxPredictionPipeline = PredictionModel.mlContext.Transforms.ApplyOnnxModel(modelFile: modelPath, outputColumnName: "dense_1", inputColumnName: "conv2d_input");
        //    var emptyDv = mlContext.Data.LoadFromEnumerable(new OnnxInput[] { });
        //    return onnxPredictionPipeline.Fit(emptyDv);
        //    return nulk
        //} 
    }
}
