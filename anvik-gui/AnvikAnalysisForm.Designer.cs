namespace anvik.gui
{
    partial class AnvikAnalysisForm
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
            this.serverGroupComboBox = new System.Windows.Forms.ComboBox();
            this.label3 = new System.Windows.Forms.Label();
            this.serverStatesTextBox = new System.Windows.Forms.TextBox();
            this.actionsTextBox = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.systemStatesTextBox = new System.Windows.Forms.TextBox();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.groupStatesTextBox = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.strictConvCheckBox = new System.Windows.Forms.CheckBox();
            this.label8 = new System.Windows.Forms.Label();
            this.viewPolicyButton = new System.Windows.Forms.Button();
            this.optimizerComboBox = new System.Windows.Forms.ComboBox();
            this.optimizeStatusTextBox = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.cancelButton = new System.Windows.Forms.Button();
            this.objectiveTextBox = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.optimizeButton = new System.Windows.Forms.Button();
            this.allowRejectCheckBox = new System.Windows.Forms.CheckBox();
            this.ignoreRevenueCheckBox = new System.Windows.Forms.CheckBox();
            this.savePolicyButton = new System.Windows.Forms.Button();
            this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.SuspendLayout();
            // 
            // serverGroupComboBox
            // 
            this.serverGroupComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.serverGroupComboBox.FormattingEnabled = true;
            this.serverGroupComboBox.Location = new System.Drawing.Point(94, 19);
            this.serverGroupComboBox.Name = "serverGroupComboBox";
            this.serverGroupComboBox.Size = new System.Drawing.Size(121, 21);
            this.serverGroupComboBox.TabIndex = 13;
            this.serverGroupComboBox.SelectedIndexChanged += new System.EventHandler(this.serverGroupComboBox_SelectedIndexChanged);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(7, 22);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(71, 13);
            this.label3.TabIndex = 12;
            this.label3.Text = "Server group:";
            // 
            // serverStatesTextBox
            // 
            this.serverStatesTextBox.Location = new System.Drawing.Point(94, 46);
            this.serverStatesTextBox.Name = "serverStatesTextBox";
            this.serverStatesTextBox.ReadOnly = true;
            this.serverStatesTextBox.Size = new System.Drawing.Size(121, 20);
            this.serverStatesTextBox.TabIndex = 11;
            // 
            // actionsTextBox
            // 
            this.actionsTextBox.Location = new System.Drawing.Point(107, 32);
            this.actionsTextBox.Name = "actionsTextBox";
            this.actionsTextBox.ReadOnly = true;
            this.actionsTextBox.Size = new System.Drawing.Size(132, 20);
            this.actionsTextBox.TabIndex = 10;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(12, 35);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(45, 13);
            this.label2.TabIndex = 9;
            this.label2.Text = "Actions:";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 9);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(75, 13);
            this.label1.TabIndex = 8;
            this.label1.Text = "System states:";
            // 
            // systemStatesTextBox
            // 
            this.systemStatesTextBox.Location = new System.Drawing.Point(107, 6);
            this.systemStatesTextBox.Name = "systemStatesTextBox";
            this.systemStatesTextBox.ReadOnly = true;
            this.systemStatesTextBox.Size = new System.Drawing.Size(132, 20);
            this.systemStatesTextBox.TabIndex = 7;
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.groupStatesTextBox);
            this.groupBox1.Controls.Add(this.serverGroupComboBox);
            this.groupBox1.Controls.Add(this.label3);
            this.groupBox1.Controls.Add(this.label5);
            this.groupBox1.Controls.Add(this.label4);
            this.groupBox1.Controls.Add(this.serverStatesTextBox);
            this.groupBox1.Location = new System.Drawing.Point(15, 58);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(224, 97);
            this.groupBox1.TabIndex = 14;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Server Group";
            // 
            // groupStatesTextBox
            // 
            this.groupStatesTextBox.Location = new System.Drawing.Point(94, 71);
            this.groupStatesTextBox.Name = "groupStatesTextBox";
            this.groupStatesTextBox.ReadOnly = true;
            this.groupStatesTextBox.Size = new System.Drawing.Size(121, 20);
            this.groupStatesTextBox.TabIndex = 17;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(8, 71);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(70, 13);
            this.label5.TabIndex = 16;
            this.label5.Text = "Group states:";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(6, 49);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(72, 13);
            this.label4.TabIndex = 15;
            this.label4.Text = "Server states:";
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.savePolicyButton);
            this.groupBox2.Controls.Add(this.strictConvCheckBox);
            this.groupBox2.Controls.Add(this.label8);
            this.groupBox2.Controls.Add(this.viewPolicyButton);
            this.groupBox2.Controls.Add(this.optimizerComboBox);
            this.groupBox2.Controls.Add(this.optimizeStatusTextBox);
            this.groupBox2.Controls.Add(this.label7);
            this.groupBox2.Controls.Add(this.cancelButton);
            this.groupBox2.Controls.Add(this.objectiveTextBox);
            this.groupBox2.Controls.Add(this.label6);
            this.groupBox2.Controls.Add(this.optimizeButton);
            this.groupBox2.Controls.Add(this.allowRejectCheckBox);
            this.groupBox2.Controls.Add(this.ignoreRevenueCheckBox);
            this.groupBox2.Location = new System.Drawing.Point(15, 161);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(224, 260);
            this.groupBox2.TabIndex = 15;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Optimization";
            // 
            // strictConvCheckBox
            // 
            this.strictConvCheckBox.AutoSize = true;
            this.strictConvCheckBox.Location = new System.Drawing.Point(14, 92);
            this.strictConvCheckBox.Name = "strictConvCheckBox";
            this.strictConvCheckBox.Size = new System.Drawing.Size(148, 17);
            this.strictConvCheckBox.TabIndex = 24;
            this.strictConvCheckBox.Text = "Check strict convergence";
            this.strictConvCheckBox.UseVisualStyleBackColor = true;
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(7, 22);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(53, 13);
            this.label8.TabIndex = 19;
            this.label8.Text = "Optimizer:";
            // 
            // viewPolicyButton
            // 
            this.viewPolicyButton.Location = new System.Drawing.Point(94, 195);
            this.viewPolicyButton.Name = "viewPolicyButton";
            this.viewPolicyButton.Size = new System.Drawing.Size(121, 23);
            this.viewPolicyButton.TabIndex = 23;
            this.viewPolicyButton.Text = "View Policy";
            this.viewPolicyButton.UseVisualStyleBackColor = true;
            this.viewPolicyButton.Click += new System.EventHandler(this.viewPolicyButton_Click);
            // 
            // optimizerComboBox
            // 
            this.optimizerComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.optimizerComboBox.FormattingEnabled = true;
            this.optimizerComboBox.Items.AddRange(new object[] {
            "CPU",
            "GPU"});
            this.optimizerComboBox.Location = new System.Drawing.Point(94, 19);
            this.optimizerComboBox.Name = "optimizerComboBox";
            this.optimizerComboBox.Size = new System.Drawing.Size(121, 21);
            this.optimizerComboBox.TabIndex = 20;
            // 
            // optimizeStatusTextBox
            // 
            this.optimizeStatusTextBox.Location = new System.Drawing.Point(96, 143);
            this.optimizeStatusTextBox.Name = "optimizeStatusTextBox";
            this.optimizeStatusTextBox.ReadOnly = true;
            this.optimizeStatusTextBox.Size = new System.Drawing.Size(119, 20);
            this.optimizeStatusTextBox.TabIndex = 22;
            this.optimizeStatusTextBox.Text = "Ready";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(10, 146);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(40, 13);
            this.label7.TabIndex = 21;
            this.label7.Text = "Status:";
            // 
            // cancelButton
            // 
            this.cancelButton.Location = new System.Drawing.Point(117, 114);
            this.cancelButton.Name = "cancelButton";
            this.cancelButton.Size = new System.Drawing.Size(98, 23);
            this.cancelButton.TabIndex = 20;
            this.cancelButton.Text = "Cancel";
            this.cancelButton.UseVisualStyleBackColor = true;
            this.cancelButton.Click += new System.EventHandler(this.cancelButton_Click);
            // 
            // objectiveTextBox
            // 
            this.objectiveTextBox.Location = new System.Drawing.Point(96, 169);
            this.objectiveTextBox.Name = "objectiveTextBox";
            this.objectiveTextBox.ReadOnly = true;
            this.objectiveTextBox.Size = new System.Drawing.Size(119, 20);
            this.objectiveTextBox.TabIndex = 19;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(10, 172);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(72, 13);
            this.label6.TabIndex = 18;
            this.label6.Text = "Minimum loss:";
            // 
            // optimizeButton
            // 
            this.optimizeButton.Location = new System.Drawing.Point(8, 114);
            this.optimizeButton.Name = "optimizeButton";
            this.optimizeButton.Size = new System.Drawing.Size(98, 23);
            this.optimizeButton.TabIndex = 2;
            this.optimizeButton.Text = "Optimize";
            this.optimizeButton.UseVisualStyleBackColor = true;
            this.optimizeButton.Click += new System.EventHandler(this.optimizeButton_Click);
            // 
            // allowRejectCheckBox
            // 
            this.allowRejectCheckBox.AutoSize = true;
            this.allowRejectCheckBox.Location = new System.Drawing.Point(14, 69);
            this.allowRejectCheckBox.Name = "allowRejectCheckBox";
            this.allowRejectCheckBox.Size = new System.Drawing.Size(157, 17);
            this.allowRejectCheckBox.TabIndex = 1;
            this.allowRejectCheckBox.Text = "Allow rejection when not full";
            this.allowRejectCheckBox.UseVisualStyleBackColor = true;
            // 
            // ignoreRevenueCheckBox
            // 
            this.ignoreRevenueCheckBox.AutoSize = true;
            this.ignoreRevenueCheckBox.Location = new System.Drawing.Point(14, 46);
            this.ignoreRevenueCheckBox.Name = "ignoreRevenueCheckBox";
            this.ignoreRevenueCheckBox.Size = new System.Drawing.Size(98, 17);
            this.ignoreRevenueCheckBox.TabIndex = 0;
            this.ignoreRevenueCheckBox.Text = "Ignore revenue";
            this.ignoreRevenueCheckBox.UseVisualStyleBackColor = true;
            // 
            // savePolicyButton
            // 
            this.savePolicyButton.Location = new System.Drawing.Point(94, 224);
            this.savePolicyButton.Name = "savePolicyButton";
            this.savePolicyButton.Size = new System.Drawing.Size(121, 23);
            this.savePolicyButton.TabIndex = 25;
            this.savePolicyButton.Text = "Save Policy";
            this.savePolicyButton.UseVisualStyleBackColor = true;
            this.savePolicyButton.Click += new System.EventHandler(this.savePolicyButton_Click);
            // 
            // AnvikAnalysisForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(250, 433);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.actionsTextBox);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.systemStatesTextBox);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "AnvikAnalysisForm";
            this.ShowInTaskbar = false;
            this.Text = "ANVIK - Problem Analysis";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.AnvikAnalysisForm_FormClosing);
            this.Load += new System.EventHandler(this.AnvikAnalysisForm_Load);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.ComboBox serverGroupComboBox;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox serverStatesTextBox;
        private System.Windows.Forms.TextBox actionsTextBox;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox systemStatesTextBox;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.TextBox groupStatesTextBox;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.CheckBox allowRejectCheckBox;
        private System.Windows.Forms.CheckBox ignoreRevenueCheckBox;
        private System.Windows.Forms.TextBox objectiveTextBox;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Button optimizeButton;
        private System.Windows.Forms.Button cancelButton;
        private System.Windows.Forms.TextBox optimizeStatusTextBox;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Button viewPolicyButton;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.ComboBox optimizerComboBox;
        private System.Windows.Forms.CheckBox strictConvCheckBox;
        private System.Windows.Forms.Button savePolicyButton;
        private System.Windows.Forms.SaveFileDialog saveFileDialog1;
    }
}