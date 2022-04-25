namespace PokemonIdentifier
{
    partial class TestForm
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.button_submit = new System.Windows.Forms.Button();
            this.textBox_imagePath = new System.Windows.Forms.TextBox();
            this.button_browse = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // button_submit
            // 
            this.button_submit.Location = new System.Drawing.Point(557, 391);
            this.button_submit.Name = "button_submit";
            this.button_submit.Size = new System.Drawing.Size(94, 29);
            this.button_submit.TabIndex = 0;
            this.button_submit.Text = "Submit";
            this.button_submit.UseVisualStyleBackColor = true;
            // 
            // textBox_imagePath
            // 
            this.textBox_imagePath.Location = new System.Drawing.Point(151, 391);
            this.textBox_imagePath.Name = "textBox_imagePath";
            this.textBox_imagePath.Size = new System.Drawing.Size(400, 27);
            this.textBox_imagePath.TabIndex = 1;
            // 
            // button_browse
            // 
            this.button_browse.Location = new System.Drawing.Point(51, 389);
            this.button_browse.Name = "button_browse";
            this.button_browse.Size = new System.Drawing.Size(94, 29);
            this.button_browse.TabIndex = 2;
            this.button_browse.Text = "Browse";
            this.button_browse.UseVisualStyleBackColor = true;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(918, 529);
            this.Controls.Add(this.button_browse);
            this.Controls.Add(this.textBox_imagePath);
            this.Controls.Add(this.button_submit);
            this.Name = "Form1";
            this.Text = "Form1";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private Button button_submit;
        private TextBox textBox_imagePath;
        private Button button_browse;
    }
}