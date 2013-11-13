namespace anvik.gui
{
    partial class AnvikPolicyViewForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
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
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.stateNumberTextBox = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.stateDescTextBox = new System.Windows.Forms.TextBox();
            this.actionDescTextBox = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.actionNumberTextBox = new System.Windows.Forms.TextBox();
            this.vmClassComboBox = new System.Windows.Forms.ComboBox();
            this.onArrivalButton = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // stateNumberTextBox
            // 
            this.stateNumberTextBox.Location = new System.Drawing.Point(53, 12);
            this.stateNumberTextBox.Name = "stateNumberTextBox";
            this.stateNumberTextBox.Size = new System.Drawing.Size(100, 20);
            this.stateNumberTextBox.TabIndex = 0;
            this.stateNumberTextBox.KeyPress += new System.Windows.Forms.KeyPressEventHandler(this.stateNumberTextBox_KeyPress);
            this.stateNumberTextBox.Validating += new System.ComponentModel.CancelEventHandler(this.stateNumberTextBox_Validating);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 15);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(35, 13);
            this.label1.TabIndex = 1;
            this.label1.Text = "State:";
            // 
            // stateDescTextBox
            // 
            this.stateDescTextBox.Font = new System.Drawing.Font("Lucida Console", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.stateDescTextBox.Location = new System.Drawing.Point(15, 38);
            this.stateDescTextBox.Multiline = true;
            this.stateDescTextBox.Name = "stateDescTextBox";
            this.stateDescTextBox.ReadOnly = true;
            this.stateDescTextBox.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.stateDescTextBox.Size = new System.Drawing.Size(317, 176);
            this.stateDescTextBox.TabIndex = 2;
            // 
            // actionDescTextBox
            // 
            this.actionDescTextBox.Font = new System.Drawing.Font("Lucida Console", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.actionDescTextBox.Location = new System.Drawing.Point(338, 38);
            this.actionDescTextBox.Multiline = true;
            this.actionDescTextBox.Name = "actionDescTextBox";
            this.actionDescTextBox.ReadOnly = true;
            this.actionDescTextBox.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.actionDescTextBox.Size = new System.Drawing.Size(404, 176);
            this.actionDescTextBox.TabIndex = 3;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(339, 15);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(43, 13);
            this.label2.TabIndex = 4;
            this.label2.Text = "Action: ";
            // 
            // actionNumberTextBox
            // 
            this.actionNumberTextBox.Location = new System.Drawing.Point(388, 12);
            this.actionNumberTextBox.Name = "actionNumberTextBox";
            this.actionNumberTextBox.ReadOnly = true;
            this.actionNumberTextBox.Size = new System.Drawing.Size(100, 20);
            this.actionNumberTextBox.TabIndex = 5;
            // 
            // vmClassComboBox
            // 
            this.vmClassComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.vmClassComboBox.FormattingEnabled = true;
            this.vmClassComboBox.Location = new System.Drawing.Point(342, 220);
            this.vmClassComboBox.Name = "vmClassComboBox";
            this.vmClassComboBox.Size = new System.Drawing.Size(146, 21);
            this.vmClassComboBox.TabIndex = 6;
            // 
            // onArrivalButton
            // 
            this.onArrivalButton.Location = new System.Drawing.Point(494, 220);
            this.onArrivalButton.Name = "onArrivalButton";
            this.onArrivalButton.Size = new System.Drawing.Size(75, 23);
            this.onArrivalButton.TabIndex = 7;
            this.onArrivalButton.Text = "Arrival";
            this.onArrivalButton.UseVisualStyleBackColor = true;
            this.onArrivalButton.Click += new System.EventHandler(this.onArrivalButton_Click);
            // 
            // AnvikPolicyViewForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(754, 252);
            this.Controls.Add(this.onArrivalButton);
            this.Controls.Add(this.vmClassComboBox);
            this.Controls.Add(this.actionNumberTextBox);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.actionDescTextBox);
            this.Controls.Add(this.stateDescTextBox);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.stateNumberTextBox);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "AnvikPolicyViewForm";
            this.ShowInTaskbar = false;
            this.Text = "ANVIK - View Policy";
            this.Load += new System.EventHandler(this.AnvikPolicyViewForm_Load);
            this.KeyDown += new System.Windows.Forms.KeyEventHandler(this.AnvikPolicyViewForm_KeyDown);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TextBox stateNumberTextBox;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox stateDescTextBox;
        private System.Windows.Forms.TextBox actionDescTextBox;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox actionNumberTextBox;
        private System.Windows.Forms.ComboBox vmClassComboBox;
        private System.Windows.Forms.Button onArrivalButton;
    }
}