using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CsvHelper.Configuration.Attributes;

namespace Database.Classes
{
    internal class UpdatedPokemonPokedex
    {
        [Index(1)]
        public int pokedexNumber { get; set; }
        [Index(2)]
        public string name { get; set; }
    }
}
