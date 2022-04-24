using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Database.Classes
{
    internal interface IPokedexCSV
    {
        public string PathToFile { get; set; }
        void IPokedexCSV(); 
    }
}
