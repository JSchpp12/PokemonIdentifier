using System;
using System.Globalization; 
using System.Threading;
using System.Drawing;
using CsvHelper;
using CsvHelper.Configuration;
using Database.Classes;
using Common; 

namespace Database
{
    public static class Pokedex
    {
        private static List<Pokemon>? pokemonStorage; 

        static Pokedex()
        {
            var config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                HasHeaderRecord = false
            }; 

            //spin up another thread to read csv file
            using (var reader = new StreamReader(".\\Datasets\\GeneralData\\pokedex.csv"))
            using (var csv = new CsvReader(reader,config))
            {
                var records = csv.GetRecords<UpdatedPokemonPokedex>();
                foreach(var record in records)
                {
                    
                }
            }
        }
    }
}