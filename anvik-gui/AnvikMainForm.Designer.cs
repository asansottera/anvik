namespace anvik.gui
{
    partial class AnvikMainForm
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(AnvikMainForm));
            this.serverGroupListBox = new System.Windows.Forms.ListBox();
            this.label4 = new System.Windows.Forms.Label();
            this.serverGroupAddButton = new System.Windows.Forms.Button();
            this.serverGroupRemoveButton = new System.Windows.Forms.Button();
            this.vmClassRemoveButton = new System.Windows.Forms.Button();
            this.vmClassAddButton = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.vmClassListBox = new System.Windows.Forms.ListBox();
            this.serverGroupGroupBox = new System.Windows.Forms.GroupBox();
            this.resourceCapacityTextBox = new System.Windows.Forms.TextBox();
            this.serverGroupCostTextBox = new System.Windows.Forms.TextBox();
            this.serverGroupCountControl = new System.Windows.Forms.NumericUpDown();
            this.serverGroupNameTextBox = new System.Windows.Forms.TextBox();
            this.resourceCapacityComboBox = new System.Windows.Forms.ComboBox();
            this.label7 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.resourceGroupBox = new System.Windows.Forms.GroupBox();
            this.resourceNameTextBox = new System.Windows.Forms.TextBox();
            this.label10 = new System.Windows.Forms.Label();
            this.removeResourceButton = new System.Windows.Forms.Button();
            this.addResourceButton = new System.Windows.Forms.Button();
            this.label12 = new System.Windows.Forms.Label();
            this.resourceListBox = new System.Windows.Forms.ListBox();
            this.vmClassGroupBox = new System.Windows.Forms.GroupBox();
            this.vmClassMuTextBox = new System.Windows.Forms.TextBox();
            this.label13 = new System.Windows.Forms.Label();
            this.vmClassLambdaTextBox = new System.Windows.Forms.TextBox();
            this.resourceRequirementTextBox = new System.Windows.Forms.TextBox();
            this.vmClassRevenueTextBox = new System.Windows.Forms.TextBox();
            this.vmClassNameTextBox = new System.Windows.Forms.TextBox();
            this.resourceRequirementComboBox = new System.Windows.Forms.ComboBox();
            this.label1 = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.label9 = new System.Windows.Forms.Label();
            this.label11 = new System.Windows.Forms.Label();
            this.toolStrip1 = new System.Windows.Forms.ToolStrip();
            this.openButton = new System.Windows.Forms.ToolStripButton();
            this.saveButton = new System.Windows.Forms.ToolStripButton();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
            this.analyzeButton = new System.Windows.Forms.ToolStripButton();
            this.serverGroupGroupBox.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.serverGroupCountControl)).BeginInit();
            this.resourceGroupBox.SuspendLayout();
            this.vmClassGroupBox.SuspendLayout();
            this.toolStrip1.SuspendLayout();
            this.SuspendLayout();
            // 
            // serverGroupListBox
            // 
            this.serverGroupListBox.FormattingEnabled = true;
            this.serverGroupListBox.Location = new System.Drawing.Point(9, 190);
            this.serverGroupListBox.Name = "serverGroupListBox";
            this.serverGroupListBox.Size = new System.Drawing.Size(156, 95);
            this.serverGroupListBox.TabIndex = 6;
            this.serverGroupListBox.SelectedIndexChanged += new System.EventHandler(this.serverGroupListBox_SelectedIndexChanged);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(6, 174);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(75, 13);
            this.label4.TabIndex = 8;
            this.label4.Text = "Server Groups";
            // 
            // serverGroupAddButton
            // 
            this.serverGroupAddButton.Location = new System.Drawing.Point(9, 291);
            this.serverGroupAddButton.Name = "serverGroupAddButton";
            this.serverGroupAddButton.Size = new System.Drawing.Size(75, 23);
            this.serverGroupAddButton.TabIndex = 10;
            this.serverGroupAddButton.Text = "Add";
            this.serverGroupAddButton.UseVisualStyleBackColor = true;
            this.serverGroupAddButton.Click += new System.EventHandler(this.serverGroupAddButton_Click);
            // 
            // serverGroupRemoveButton
            // 
            this.serverGroupRemoveButton.Enabled = false;
            this.serverGroupRemoveButton.Location = new System.Drawing.Point(90, 291);
            this.serverGroupRemoveButton.Name = "serverGroupRemoveButton";
            this.serverGroupRemoveButton.Size = new System.Drawing.Size(75, 23);
            this.serverGroupRemoveButton.TabIndex = 11;
            this.serverGroupRemoveButton.Text = "Remove";
            this.serverGroupRemoveButton.UseVisualStyleBackColor = true;
            this.serverGroupRemoveButton.Click += new System.EventHandler(this.serverGroupRemoveButton_Click);
            // 
            // vmClassRemoveButton
            // 
            this.vmClassRemoveButton.Enabled = false;
            this.vmClassRemoveButton.Location = new System.Drawing.Point(90, 436);
            this.vmClassRemoveButton.Name = "vmClassRemoveButton";
            this.vmClassRemoveButton.Size = new System.Drawing.Size(75, 23);
            this.vmClassRemoveButton.TabIndex = 15;
            this.vmClassRemoveButton.Text = "Remove";
            this.vmClassRemoveButton.UseVisualStyleBackColor = true;
            this.vmClassRemoveButton.Click += new System.EventHandler(this.vmClassRemoveButton_Click);
            // 
            // vmClassAddButton
            // 
            this.vmClassAddButton.Location = new System.Drawing.Point(9, 436);
            this.vmClassAddButton.Name = "vmClassAddButton";
            this.vmClassAddButton.Size = new System.Drawing.Size(75, 23);
            this.vmClassAddButton.TabIndex = 14;
            this.vmClassAddButton.Text = "Add";
            this.vmClassAddButton.UseVisualStyleBackColor = true;
            this.vmClassAddButton.Click += new System.EventHandler(this.vmClassAddButton_Click);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(6, 319);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(119, 13);
            this.label2.TabIndex = 13;
            this.label2.Text = "Virtual Machine Classes";
            // 
            // vmClassListBox
            // 
            this.vmClassListBox.FormattingEnabled = true;
            this.vmClassListBox.Location = new System.Drawing.Point(9, 335);
            this.vmClassListBox.Name = "vmClassListBox";
            this.vmClassListBox.Size = new System.Drawing.Size(156, 95);
            this.vmClassListBox.TabIndex = 12;
            this.vmClassListBox.SelectedIndexChanged += new System.EventHandler(this.vmClassListBox_SelectedIndexChanged);
            // 
            // serverGroupGroupBox
            // 
            this.serverGroupGroupBox.Controls.Add(this.resourceCapacityTextBox);
            this.serverGroupGroupBox.Controls.Add(this.serverGroupCostTextBox);
            this.serverGroupGroupBox.Controls.Add(this.serverGroupCountControl);
            this.serverGroupGroupBox.Controls.Add(this.serverGroupNameTextBox);
            this.serverGroupGroupBox.Controls.Add(this.resourceCapacityComboBox);
            this.serverGroupGroupBox.Controls.Add(this.label7);
            this.serverGroupGroupBox.Controls.Add(this.label6);
            this.serverGroupGroupBox.Controls.Add(this.label5);
            this.serverGroupGroupBox.Controls.Add(this.label3);
            this.serverGroupGroupBox.Enabled = false;
            this.serverGroupGroupBox.Location = new System.Drawing.Point(171, 190);
            this.serverGroupGroupBox.Name = "serverGroupGroupBox";
            this.serverGroupGroupBox.Size = new System.Drawing.Size(385, 124);
            this.serverGroupGroupBox.TabIndex = 16;
            this.serverGroupGroupBox.TabStop = false;
            this.serverGroupGroupBox.Text = "Server group";
            // 
            // resourceCapacityTextBox
            // 
            this.resourceCapacityTextBox.Enabled = false;
            this.resourceCapacityTextBox.Location = new System.Drawing.Point(213, 98);
            this.resourceCapacityTextBox.Name = "resourceCapacityTextBox";
            this.resourceCapacityTextBox.Size = new System.Drawing.Size(120, 20);
            this.resourceCapacityTextBox.TabIndex = 19;
            this.resourceCapacityTextBox.Leave += new System.EventHandler(this.resourceCapacityTextBox_Leave);
            // 
            // serverGroupCostTextBox
            // 
            this.serverGroupCostTextBox.Location = new System.Drawing.Point(86, 72);
            this.serverGroupCostTextBox.Name = "serverGroupCostTextBox";
            this.serverGroupCostTextBox.Size = new System.Drawing.Size(120, 20);
            this.serverGroupCostTextBox.TabIndex = 18;
            this.serverGroupCostTextBox.Leave += new System.EventHandler(this.serverGroupCostTextBox_Leave);
            // 
            // serverGroupCountControl
            // 
            this.serverGroupCountControl.Location = new System.Drawing.Point(86, 45);
            this.serverGroupCountControl.Name = "serverGroupCountControl";
            this.serverGroupCountControl.Size = new System.Drawing.Size(120, 20);
            this.serverGroupCountControl.TabIndex = 17;
            this.serverGroupCountControl.Leave += new System.EventHandler(this.serverGroupCountControl_Leave);
            // 
            // serverGroupNameTextBox
            // 
            this.serverGroupNameTextBox.Location = new System.Drawing.Point(86, 15);
            this.serverGroupNameTextBox.Name = "serverGroupNameTextBox";
            this.serverGroupNameTextBox.Size = new System.Drawing.Size(120, 20);
            this.serverGroupNameTextBox.TabIndex = 5;
            this.serverGroupNameTextBox.Leave += new System.EventHandler(this.serverGroupNameTextBox_Leave);
            // 
            // resourceCapacityComboBox
            // 
            this.resourceCapacityComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.resourceCapacityComboBox.FormattingEnabled = true;
            this.resourceCapacityComboBox.Location = new System.Drawing.Point(86, 97);
            this.resourceCapacityComboBox.Name = "resourceCapacityComboBox";
            this.resourceCapacityComboBox.Size = new System.Drawing.Size(121, 21);
            this.resourceCapacityComboBox.TabIndex = 4;
            this.resourceCapacityComboBox.SelectedIndexChanged += new System.EventHandler(this.resourceCapacityComboBox_SelectedIndexChanged);
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(6, 100);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(48, 13);
            this.label7.TabIndex = 3;
            this.label7.Text = "Capacity";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(8, 74);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(28, 13);
            this.label6.TabIndex = 2;
            this.label6.Text = "Cost";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(7, 47);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(27, 13);
            this.label5.TabIndex = 1;
            this.label5.Text = "Size";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(7, 22);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(35, 13);
            this.label3.TabIndex = 0;
            this.label3.Text = "Name";
            // 
            // resourceGroupBox
            // 
            this.resourceGroupBox.Controls.Add(this.resourceNameTextBox);
            this.resourceGroupBox.Controls.Add(this.label10);
            this.resourceGroupBox.Enabled = false;
            this.resourceGroupBox.Location = new System.Drawing.Point(171, 45);
            this.resourceGroupBox.Name = "resourceGroupBox";
            this.resourceGroupBox.Size = new System.Drawing.Size(385, 124);
            this.resourceGroupBox.TabIndex = 25;
            this.resourceGroupBox.TabStop = false;
            this.resourceGroupBox.Text = "Resource";
            // 
            // resourceNameTextBox
            // 
            this.resourceNameTextBox.Location = new System.Drawing.Point(86, 19);
            this.resourceNameTextBox.Name = "resourceNameTextBox";
            this.resourceNameTextBox.Size = new System.Drawing.Size(120, 20);
            this.resourceNameTextBox.TabIndex = 5;
            this.resourceNameTextBox.Leave += new System.EventHandler(this.resourceNameTextBox_Leave);
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(7, 22);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(35, 13);
            this.label10.TabIndex = 0;
            this.label10.Text = "Name";
            // 
            // removeResourceButton
            // 
            this.removeResourceButton.Enabled = false;
            this.removeResourceButton.Location = new System.Drawing.Point(90, 146);
            this.removeResourceButton.Name = "removeResourceButton";
            this.removeResourceButton.Size = new System.Drawing.Size(75, 23);
            this.removeResourceButton.TabIndex = 23;
            this.removeResourceButton.Text = "Remove";
            this.removeResourceButton.UseVisualStyleBackColor = true;
            this.removeResourceButton.Click += new System.EventHandler(this.removeResourceButton_Click);
            // 
            // addResourceButton
            // 
            this.addResourceButton.Location = new System.Drawing.Point(9, 146);
            this.addResourceButton.Name = "addResourceButton";
            this.addResourceButton.Size = new System.Drawing.Size(75, 23);
            this.addResourceButton.TabIndex = 22;
            this.addResourceButton.Text = "Add";
            this.addResourceButton.UseVisualStyleBackColor = true;
            this.addResourceButton.Click += new System.EventHandler(this.addResourceButton_Click);
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Location = new System.Drawing.Point(6, 29);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(58, 13);
            this.label12.TabIndex = 21;
            this.label12.Text = "Resources";
            // 
            // resourceListBox
            // 
            this.resourceListBox.FormattingEnabled = true;
            this.resourceListBox.Location = new System.Drawing.Point(9, 45);
            this.resourceListBox.Name = "resourceListBox";
            this.resourceListBox.Size = new System.Drawing.Size(156, 95);
            this.resourceListBox.TabIndex = 20;
            this.resourceListBox.SelectedIndexChanged += new System.EventHandler(this.resourceListBox_SelectedIndexChanged);
            // 
            // vmClassGroupBox
            // 
            this.vmClassGroupBox.Controls.Add(this.vmClassMuTextBox);
            this.vmClassGroupBox.Controls.Add(this.label13);
            this.vmClassGroupBox.Controls.Add(this.vmClassLambdaTextBox);
            this.vmClassGroupBox.Controls.Add(this.resourceRequirementTextBox);
            this.vmClassGroupBox.Controls.Add(this.vmClassRevenueTextBox);
            this.vmClassGroupBox.Controls.Add(this.vmClassNameTextBox);
            this.vmClassGroupBox.Controls.Add(this.resourceRequirementComboBox);
            this.vmClassGroupBox.Controls.Add(this.label1);
            this.vmClassGroupBox.Controls.Add(this.label8);
            this.vmClassGroupBox.Controls.Add(this.label9);
            this.vmClassGroupBox.Controls.Add(this.label11);
            this.vmClassGroupBox.Enabled = false;
            this.vmClassGroupBox.Location = new System.Drawing.Point(171, 335);
            this.vmClassGroupBox.Name = "vmClassGroupBox";
            this.vmClassGroupBox.Size = new System.Drawing.Size(385, 162);
            this.vmClassGroupBox.TabIndex = 20;
            this.vmClassGroupBox.TabStop = false;
            this.vmClassGroupBox.Text = "Server group";
            // 
            // vmClassMuTextBox
            // 
            this.vmClassMuTextBox.Location = new System.Drawing.Point(86, 72);
            this.vmClassMuTextBox.Name = "vmClassMuTextBox";
            this.vmClassMuTextBox.Size = new System.Drawing.Size(120, 20);
            this.vmClassMuTextBox.TabIndex = 22;
            this.vmClassMuTextBox.Leave += new System.EventHandler(this.vmClassMuTextBox_Leave);
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.Location = new System.Drawing.Point(7, 75);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(64, 13);
            this.label13.TabIndex = 21;
            this.label13.Text = "Service rate";
            // 
            // vmClassLambdaTextBox
            // 
            this.vmClassLambdaTextBox.Location = new System.Drawing.Point(86, 47);
            this.vmClassLambdaTextBox.Name = "vmClassLambdaTextBox";
            this.vmClassLambdaTextBox.Size = new System.Drawing.Size(120, 20);
            this.vmClassLambdaTextBox.TabIndex = 20;
            this.vmClassLambdaTextBox.Leave += new System.EventHandler(this.vmClassLambdaTextBox_Leave);
            // 
            // resourceRequirementTextBox
            // 
            this.resourceRequirementTextBox.Enabled = false;
            this.resourceRequirementTextBox.Location = new System.Drawing.Point(213, 125);
            this.resourceRequirementTextBox.Name = "resourceRequirementTextBox";
            this.resourceRequirementTextBox.Size = new System.Drawing.Size(120, 20);
            this.resourceRequirementTextBox.TabIndex = 19;
            this.resourceRequirementTextBox.Leave += new System.EventHandler(this.resourceRequirementTextBox_Leave);
            // 
            // vmClassRevenueTextBox
            // 
            this.vmClassRevenueTextBox.Location = new System.Drawing.Point(86, 98);
            this.vmClassRevenueTextBox.Name = "vmClassRevenueTextBox";
            this.vmClassRevenueTextBox.Size = new System.Drawing.Size(120, 20);
            this.vmClassRevenueTextBox.TabIndex = 18;
            this.vmClassRevenueTextBox.Leave += new System.EventHandler(this.vmClassRevenueTextBox_Leave);
            // 
            // vmClassNameTextBox
            // 
            this.vmClassNameTextBox.Location = new System.Drawing.Point(86, 19);
            this.vmClassNameTextBox.Name = "vmClassNameTextBox";
            this.vmClassNameTextBox.Size = new System.Drawing.Size(120, 20);
            this.vmClassNameTextBox.TabIndex = 5;
            this.vmClassNameTextBox.Leave += new System.EventHandler(this.vmClassNameTextBox_Leave);
            // 
            // resourceRequirementComboBox
            // 
            this.resourceRequirementComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.resourceRequirementComboBox.FormattingEnabled = true;
            this.resourceRequirementComboBox.Location = new System.Drawing.Point(86, 124);
            this.resourceRequirementComboBox.Name = "resourceRequirementComboBox";
            this.resourceRequirementComboBox.Size = new System.Drawing.Size(121, 21);
            this.resourceRequirementComboBox.TabIndex = 4;
            this.resourceRequirementComboBox.SelectedIndexChanged += new System.EventHandler(this.resourceRequirementComboBox_SelectedIndexChanged);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(6, 127);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(67, 13);
            this.label1.TabIndex = 3;
            this.label1.Text = "Requirement";
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(8, 101);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(51, 13);
            this.label8.TabIndex = 2;
            this.label8.Text = "Revenue";
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(7, 50);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(57, 13);
            this.label9.TabIndex = 1;
            this.label9.Text = "Arrival rate";
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Location = new System.Drawing.Point(7, 22);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(35, 13);
            this.label11.TabIndex = 0;
            this.label11.Text = "Name";
            // 
            // toolStrip1
            // 
            this.toolStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.openButton,
            this.saveButton,
            this.analyzeButton});
            this.toolStrip1.Location = new System.Drawing.Point(0, 0);
            this.toolStrip1.Name = "toolStrip1";
            this.toolStrip1.Size = new System.Drawing.Size(568, 25);
            this.toolStrip1.TabIndex = 26;
            this.toolStrip1.Text = "toolStrip1";
            // 
            // openButton
            // 
            this.openButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.openButton.Image = ((System.Drawing.Image)(resources.GetObject("openButton.Image")));
            this.openButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.openButton.Name = "openButton";
            this.openButton.Size = new System.Drawing.Size(40, 22);
            this.openButton.Text = "Open";
            this.openButton.Click += new System.EventHandler(this.openButton_Click);
            // 
            // saveButton
            // 
            this.saveButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.saveButton.Image = ((System.Drawing.Image)(resources.GetObject("saveButton.Image")));
            this.saveButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.saveButton.Name = "saveButton";
            this.saveButton.Size = new System.Drawing.Size(35, 22);
            this.saveButton.Text = "Save";
            this.saveButton.Click += new System.EventHandler(this.saveButton_Click);
            // 
            // analyzeButton
            // 
            this.analyzeButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.analyzeButton.Image = ((System.Drawing.Image)(resources.GetObject("analyzeButton.Image")));
            this.analyzeButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.analyzeButton.Name = "analyzeButton";
            this.analyzeButton.Size = new System.Drawing.Size(52, 22);
            this.analyzeButton.Text = "Analyze";
            this.analyzeButton.Click += new System.EventHandler(this.analyzeButton_Click);
            // 
            // AnvikMainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(568, 509);
            this.Controls.Add(this.toolStrip1);
            this.Controls.Add(this.vmClassGroupBox);
            this.Controls.Add(this.resourceGroupBox);
            this.Controls.Add(this.serverGroupGroupBox);
            this.Controls.Add(this.removeResourceButton);
            this.Controls.Add(this.vmClassRemoveButton);
            this.Controls.Add(this.addResourceButton);
            this.Controls.Add(this.vmClassAddButton);
            this.Controls.Add(this.label12);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.resourceListBox);
            this.Controls.Add(this.vmClassListBox);
            this.Controls.Add(this.serverGroupRemoveButton);
            this.Controls.Add(this.serverGroupAddButton);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.serverGroupListBox);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.MaximizeBox = false;
            this.Name = "AnvikMainForm";
            this.Text = "ANVIK - Optimal virtual machine scheduling";
            this.Load += new System.EventHandler(this.Form1_Load);
            this.serverGroupGroupBox.ResumeLayout(false);
            this.serverGroupGroupBox.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.serverGroupCountControl)).EndInit();
            this.resourceGroupBox.ResumeLayout(false);
            this.resourceGroupBox.PerformLayout();
            this.vmClassGroupBox.ResumeLayout(false);
            this.vmClassGroupBox.PerformLayout();
            this.toolStrip1.ResumeLayout(false);
            this.toolStrip1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.ListBox serverGroupListBox;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Button serverGroupAddButton;
        private System.Windows.Forms.Button serverGroupRemoveButton;
        private System.Windows.Forms.Button vmClassRemoveButton;
        private System.Windows.Forms.Button vmClassAddButton;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.ListBox vmClassListBox;
        private System.Windows.Forms.GroupBox serverGroupGroupBox;
        private System.Windows.Forms.NumericUpDown serverGroupCountControl;
        private System.Windows.Forms.TextBox serverGroupNameTextBox;
        private System.Windows.Forms.ComboBox resourceCapacityComboBox;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.GroupBox resourceGroupBox;
        private System.Windows.Forms.TextBox resourceNameTextBox;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.Button removeResourceButton;
        private System.Windows.Forms.Button addResourceButton;
        private System.Windows.Forms.Label label12;
        private System.Windows.Forms.ListBox resourceListBox;
        private System.Windows.Forms.GroupBox vmClassGroupBox;
        private System.Windows.Forms.TextBox vmClassNameTextBox;
        private System.Windows.Forms.ComboBox resourceRequirementComboBox;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.Label label13;
        private System.Windows.Forms.ToolStrip toolStrip1;
        private System.Windows.Forms.ToolStripButton openButton;
        private System.Windows.Forms.ToolStripButton saveButton;
        private System.Windows.Forms.TextBox resourceCapacityTextBox;
        private System.Windows.Forms.TextBox serverGroupCostTextBox;
        private System.Windows.Forms.TextBox vmClassMuTextBox;
        private System.Windows.Forms.TextBox vmClassLambdaTextBox;
        private System.Windows.Forms.TextBox resourceRequirementTextBox;
        private System.Windows.Forms.TextBox vmClassRevenueTextBox;
        private System.Windows.Forms.OpenFileDialog openFileDialog1;
        private System.Windows.Forms.SaveFileDialog saveFileDialog1;
        private System.Windows.Forms.ToolStripButton analyzeButton;
    }
}

