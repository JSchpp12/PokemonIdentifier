using Models;

namespace PokemonIdentifier
{
    public partial class TestForm : Form
    {
        private string selectedFilePath = string.Empty;

        public TestForm()
        {
            InitializeComponent();

            //load image 
            pictureBox_pokemon.Load("./TestImages/Bulbasaur.png");
            pictureBox_pokemon.SizeMode = PictureBoxSizeMode.StretchImage; 
        }

        private void button_submit_Click(object sender, EventArgs e)
        {
            //read image and pass it off to the web helper 
            if (selectedFilePath != string.Empty) 
            {
                Image imageToTest = Image.FromFile(selectedFilePath); 
                
            }

        }

        private void button_browse_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog ofd = new OpenFileDialog())
            {
                ofd.InitialDirectory = "c:\\";
                ofd.Filter = "Image Files (*.jpg)|*jpg";
                ofd.RestoreDirectory = true; 

                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    //get path of specified file 
                    selectedFilePath = ofd.FileName;
                    textBox_imagePath.Text = selectedFilePath;
                }
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Bitmap testImage; 
            //test
            //
            //string[] list = new string[] { "./TestImages/gen6--abomasnow.png" };
            string[] list = new string[] { "./TestImages/test.png" };

            foreach (string item in list)
            {
                var bitmap = new Bitmap(list[0]); 


                //using(var stream = new FileStream(item, FileMode.Open))
                //{
                //    testImage = (Bitmap)Image.FromStream(stream);

                //    testImage.MakeTransparent(); 

                //}
                Models.MachineLearning.PredictionModel.ClassifyImage(bitmap); 
            }
        }
    }
}