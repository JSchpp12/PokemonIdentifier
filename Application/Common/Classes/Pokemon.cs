using System.Drawing; 

namespace Common
{
    public class Pokemon
    {
        public string Name { get; set; }
        public int PokedexID { get; set; }
        public string ImagePath { get; set; }

        public Pokemon(string name, int pokedexID)
        {
            Name = name;
            PokedexID = pokedexID;
        }
    }
}