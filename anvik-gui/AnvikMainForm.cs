using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Xml.Serialization;
using anvik;

namespace anvik.gui
{
    public partial class AnvikMainForm : Form
    {
        private Problem p;
        private AnvikAnalysisForm analysisForm;

        public AnvikMainForm()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            p = new Problem();
            analysisForm = new AnvikAnalysisForm();
        }

        void reset()
        {
            // reset UI
            resourceListBox.Items.Clear();
            resourceCapacityComboBox.Items.Clear();
            resourceRequirementComboBox.Items.Clear();
            foreach (String resource in p.Resources)
            {
                resourceListBox.Items.Add(resource);
                resourceCapacityComboBox.Items.Add(resource);
                resourceRequirementComboBox.Items.Add(resource);
            }
            resourceNameTextBox.Text = "";
            resourceGroupBox.Enabled = false;
            serverGroupListBox.Items.Clear();
            foreach (ServerGroup sg in p.ServerGroups)
            {
                serverGroupListBox.Items.Add(sg.Name);
            }
            serverGroupNameTextBox.Text = "";
            serverGroupCountControl.Value = 0;
            serverGroupCostTextBox.Text = "";
            resourceCapacityTextBox.Text = "";
            serverGroupGroupBox.Enabled = false;
            vmClassListBox.Items.Clear();
            foreach (VmClass vmc in p.VmClasses)
            {
                vmClassListBox.Items.Add(vmc.Name);
            }
            vmClassNameTextBox.Text = "";
            vmClassLambdaTextBox.Text = "";
            vmClassMuTextBox.Text = "";
            vmClassRevenueTextBox.Text = "";
            resourceRequirementTextBox.Text = "";
            vmClassGroupBox.Enabled = false;
        }

        private void ShowNotANumber()
        {
            MessageBox.Show("Not a floating point number", "Input Error", MessageBoxButtons.OK);
        }

        private void addResourceButton_Click(object sender, EventArgs e)
        {
            String resourceName = "Resource " + (p.Resources.Count + 1);
            p.Resources.Add(resourceName);
            resourceListBox.Items.Add(resourceName);
            resourceCapacityComboBox.Items.Add(resourceName);
            resourceRequirementComboBox.Items.Add(resourceName);
            foreach (ServerGroup sg in p.ServerGroups)
            {
                sg.Capacity.Add(1);
            }
            foreach (VmClass vmc in p.VmClasses)
            {
                vmc.Requirement.Add(1);
            }
        }

        private void removeResourceButton_Click(object sender, EventArgs e)
        {
            int idx = resourceListBox.SelectedIndex;
            if (idx >= 0)
            {
                p.Resources.RemoveAt(idx);
                resourceListBox.Items.RemoveAt(idx);
                resourceCapacityComboBox.Items.RemoveAt(idx);
                resourceRequirementComboBox.Items.RemoveAt(idx);
                if (resourceCapacityComboBox.SelectedIndex < 0)
                {
                    resourceCapacityTextBox.Text = "";
                    resourceCapacityTextBox.Enabled = false;
                }
                if (resourceRequirementComboBox.SelectedIndex < 0)
                {
                    resourceRequirementTextBox.Text = "";
                    resourceRequirementTextBox.Enabled = false;
                }
                foreach (ServerGroup sg in p.ServerGroups)
                {
                    sg.Capacity.RemoveAt(idx);
                }
                foreach (VmClass vmc in p.VmClasses)
                {
                    vmc.Requirement.RemoveAt(idx);
                }
            }
        }

        private void resourceListBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            bool selected = (resourceListBox.SelectedIndex >= 0);
            removeResourceButton.Enabled = selected;
            resourceGroupBox.Enabled = selected;
            if (selected)
            {
                int idx = resourceListBox.SelectedIndex;
                resourceNameTextBox.Text = p.Resources[idx];
            } else {
                resourceNameTextBox.Text = "";
            }
        }

        private void resourceNameTextBox_Leave(object sender, EventArgs e)
        {
            int idx = resourceListBox.SelectedIndex;
            if (idx >= 0)
            {
                String name = resourceNameTextBox.Text;
                p.Resources[idx] = name;
                resourceListBox.Items[idx] = name;
                resourceCapacityComboBox.Items[idx] = name;
                resourceRequirementComboBox.Items[idx] = name;
            }
        }

        private void serverGroupAddButton_Click(object sender, EventArgs e)
        {
            // update data
            ServerGroup sg = new ServerGroup();
            sg.Name = "Servers " + (p.ServerGroups.Count + 1);
            sg.Count = 1;
            sg.Cost = 0.0f;
            foreach (String r in p.Resources) {
                sg.Capacity.Add(1);
            }
            p.ServerGroups.Add(sg);
            // update list
            serverGroupListBox.Items.Add(sg.Name);
        }

        private void serverGroupRemoveButton_Click(object sender, EventArgs e)
        {
            int idx = serverGroupListBox.SelectedIndex;
            if (idx >= 0)
            {
                p.ServerGroups.RemoveAt(idx);
                serverGroupListBox.Items.RemoveAt(idx);
            }
        }

        private void serverGroupListBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            bool selected = (serverGroupListBox.SelectedIndex >= 0);
            serverGroupRemoveButton.Enabled = selected;
            serverGroupGroupBox.Enabled = selected;
            if (selected)
            {
                int idx = serverGroupListBox.SelectedIndex;
                ServerGroup sg = p.ServerGroups[idx];
                serverGroupNameTextBox.Text = sg.Name;
                serverGroupCountControl.Value = sg.Count;
                serverGroupCostTextBox.Text = sg.Cost.ToString();
                int rIdx = resourceCapacityComboBox.SelectedIndex;
                if (rIdx >= 0)
                {
                    resourceCapacityTextBox.Text = sg.Capacity[rIdx].ToString();
                }
            }
            else
            {
                serverGroupNameTextBox.Text = "";
                serverGroupCountControl.Value = 0;
                serverGroupCostTextBox.Text = "";
                resourceCapacityTextBox.Text = "";
            }
        }

        private void serverGroupNameTextBox_Leave(object sender, EventArgs e)
        {
            int idx = serverGroupListBox.SelectedIndex;
            if (idx >= 0)
            {
                String name = serverGroupNameTextBox.Text;
                p.ServerGroups[idx].Name = name;
                serverGroupListBox.Items[idx] = name;
            }
        }

        private void serverGroupCountControl_Leave(object sender, EventArgs e)
        {
            int idx = serverGroupListBox.SelectedIndex;
            if (idx >= 0)
            {
                uint count = (uint)serverGroupCountControl.Value;
                p.ServerGroups[idx].Count = count;
            }
        }

        private void serverGroupCostTextBox_Leave(object sender, EventArgs e)
        {
            int idx = serverGroupListBox.SelectedIndex;
            if (idx >= 0)
            {
                float cost;
                if (Single.TryParse(serverGroupCostTextBox.Text, out cost))
                {
                    p.ServerGroups[idx].Cost = cost;
                }
                else
                {
                    ShowNotANumber();
                    serverGroupCostTextBox.Focus();
                }
                
            }
        }

        private void vmClassAddButton_Click(object sender, EventArgs e)
        {
            // update data
            VmClass vmc = new VmClass();
            vmc.Name = "Virtual Machines " + (p.VmClasses.Count + 1);
            vmc.ArrivalRate = 1.0f;
            vmc.ServiceRate = 2.0f;
            vmc.Revenue = 0.0f;
            foreach (String r in p.Resources)
            {
                vmc.Requirement.Add(1);
            }
            p.VmClasses.Add(vmc);
            // update list
            vmClassListBox.Items.Add(vmc.Name);
        }

        private void vmClassListBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            bool selected = (vmClassListBox.SelectedIndex >= 0);
            vmClassRemoveButton.Enabled = selected;
            vmClassGroupBox.Enabled = selected;
            if (selected)
            {
                int idx = vmClassListBox.SelectedIndex;
                VmClass vmc = p.VmClasses[idx];
                vmClassNameTextBox.Text = vmc.Name;
                vmClassLambdaTextBox.Text = vmc.ArrivalRate.ToString();
                vmClassMuTextBox.Text = vmc.ServiceRate.ToString();
                vmClassRevenueTextBox.Text = vmc.Revenue.ToString();
                int rIdx = resourceRequirementComboBox.SelectedIndex;
                if (rIdx >= 0)
                {
                    resourceRequirementTextBox.Text = vmc.Requirement[rIdx].ToString();
                }
            }
            else
            {
                vmClassNameTextBox.Text = "";
                vmClassLambdaTextBox.Text = "";
                vmClassMuTextBox.Text = "";
                vmClassRevenueTextBox.Text = "";
                resourceRequirementTextBox.Text = "";
            }
        }

        private void vmClassRemoveButton_Click(object sender, EventArgs e)
        {
            int idx = vmClassListBox.SelectedIndex;
            if (idx >= 0)
            {
                p.VmClasses.RemoveAt(idx);
                vmClassListBox.Items.RemoveAt(idx);
            }
        }

        private void openButton_Click(object sender, EventArgs e)
        {
            openFileDialog1.Filter = "Anvik Problem File|*.akp";
            DialogResult res = openFileDialog1.ShowDialog();
            if (res == System.Windows.Forms.DialogResult.OK)
            {
                try
                {
                    p.Load(openFileDialog1.FileName);
                    reset();
                }
                catch (ProblemLoadException ex)
                {
                    MessageBox.Show(this, ex.Message, "Load Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }

        private void saveButton_Click(object sender, EventArgs e)
        {
            List<String> messages = new List<String>();
            if (p.Check(messages))
            {
                saveFileDialog1.Filter = "Anvik Problem File|*.akp";
                DialogResult res = saveFileDialog1.ShowDialog();
                if (res == DialogResult.OK)
                {
                    try
                    {
                        p.Save(saveFileDialog1.FileName);
                        reset();
                    }
                    catch (ProblemSaveException ex)
                    {
                        MessageBox.Show(this, ex.Message, "Save Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                }
            }
            else
            {
                MessageBox.Show(messages.Count > 0 ? messages[0] : "Unspecified error", "Invalid Problem", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
            }
        }

        private void resourceCapacityComboBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            int sgIdx = serverGroupListBox.SelectedIndex;
            int idx = resourceCapacityComboBox.SelectedIndex;
            if (idx >= 0 && sgIdx >= 0)
            {
                ServerGroup sg = p.ServerGroups[sgIdx];
                resourceCapacityTextBox.Enabled = true;
                resourceCapacityTextBox.Text = sg.Capacity[idx].ToString();
            }
            else
            {
                resourceCapacityTextBox.Enabled = true;
                resourceCapacityTextBox.Text = "";
            }
        }

        private void resourceRequirementComboBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            int vmcIdx = vmClassListBox.SelectedIndex;
            int idx = resourceRequirementComboBox.SelectedIndex;
            if (idx >= 0 && vmcIdx >= 0)
            {
                VmClass vmc = p.VmClasses[vmcIdx];
                resourceRequirementTextBox.Enabled = true;
                resourceRequirementTextBox.Text = vmc.Requirement[idx].ToString();
            }
            else
            {
                resourceRequirementTextBox.Enabled = true;
                resourceRequirementTextBox.Text = "";
            }
        }

        private void resourceCapacityTextBox_Leave(object sender, EventArgs e)
        {
            int sgIdx = serverGroupListBox.SelectedIndex;
            int idx = resourceCapacityComboBox.SelectedIndex;
            if (idx >= 0 && sgIdx >= 0)
            {
                uint capacity;
                if (UInt32.TryParse(resourceCapacityTextBox.Text, out capacity))
                {
                    p.ServerGroups[sgIdx].Capacity[idx] = capacity;
                }
                else
                {
                    ShowNotANumber();
                    resourceCapacityTextBox.Focus();
                }
            }
        }

        private void resourceRequirementTextBox_Leave(object sender, EventArgs e)
        {
            int vmcIdx = vmClassListBox.SelectedIndex;
            int idx = resourceRequirementComboBox.SelectedIndex;
            if (idx >= 0 && vmcIdx >= 0)
            {
                uint requirement;
                if (UInt32.TryParse(resourceRequirementTextBox.Text, out requirement))
                {
                    p.VmClasses[vmcIdx].Requirement[idx] = requirement;
                }
                else
                {
                    ShowNotANumber();
                    resourceRequirementTextBox.Focus();
                }
            }
        }

        private void vmClassNameTextBox_Leave(object sender, EventArgs e)
        {
            int idx = vmClassListBox.SelectedIndex;
            if (idx >= 0)
            {
                String name = vmClassNameTextBox.Text;
                p.VmClasses[idx].Name = name;
                vmClassListBox.Items[idx] = name;
            }
        }

        private void vmClassLambdaTextBox_Leave(object sender, EventArgs e)
        {
            int idx = vmClassListBox.SelectedIndex;
            if (idx >= 0)
            {
                float arrivalRate;
                if (Single.TryParse(vmClassLambdaTextBox.Text, out arrivalRate))
                {
                    p.VmClasses[idx].ArrivalRate = arrivalRate;
                }
                else
                {
                    ShowNotANumber();
                    vmClassLambdaTextBox.Focus();
                }
            }
        }

        private void vmClassMuTextBox_Leave(object sender, EventArgs e)
        {
            int idx = vmClassListBox.SelectedIndex;
            if (idx >= 0)
            {
                float serviceRate;
                if (Single.TryParse(vmClassMuTextBox.Text, out serviceRate))
                {
                    p.VmClasses[idx].ServiceRate = serviceRate;
                }
                else
                {
                    ShowNotANumber();
                    vmClassMuTextBox.Focus();
                }
            }
        }

        private void vmClassRevenueTextBox_Leave(object sender, EventArgs e)
        {
            int idx = vmClassListBox.SelectedIndex;
            if (idx >= 0)
            {
                float revenue;
                if (Single.TryParse(vmClassRevenueTextBox.Text, out revenue))
                {
                    p.VmClasses[idx].Revenue = revenue;
                }
                else
                {
                    ShowNotANumber();
                    vmClassRevenueTextBox.Focus();
                }
            }
        }

        private void analyzeButton_Click(object sender, EventArgs e)
        {
            List<String> messages = new List<String>();
            if (p.Check(messages))
            {
                try
                {
                    Analysis analysis = new Analysis(p);
                    analysisForm.Problem = p;
                    analysisForm.Analysis = analysis;
                    analysisForm.ShowDialog(this);
                }
                catch (TooManyStatesException ex)
                {
                    MessageBox.Show(ex.Message, "Too many states", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                }
            }
            else
            {
                MessageBox.Show(messages.Count > 0 ? messages[0] : "Unspecified error", "Invalid Problem", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
            }
        }
    }
}
