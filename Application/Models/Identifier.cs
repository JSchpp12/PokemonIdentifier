using System;
using System.Threading;
using Tensorflow;
using NumSharp; 

namespace Models
{

    public static class Identifier
    {
        static string pathToModelFile = @".\Models\Identifier\saved_model.pb";
 
        //initalize the tensorflow model 
        static Identifier()
        {
            byte[] buffer = System.IO.File.ReadAllBytes(pathToModelFile);

            var graph = Graph.ImportFromPB(pathToModelFile); 
        }
    }
}