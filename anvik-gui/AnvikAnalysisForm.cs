using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace anvik.gui
{
    public partial class AnvikAnalysisForm : Form
    {
        public AnvikAnalysisForm()
        {
            InitializeComponent();
        }

        private Analysis analysis;
        private Problem problem;
        private bool optimizationRunning;
        private bool optimizationCanceled;
        AnvikPolicyViewForm policyViewForm = new AnvikPolicyViewForm();

        public Analysis Analysis
        {
            get { return analysis; }
            set { analysis = value; }
        }

        public Problem Problem
        {
            get { return problem; }
            set { problem = value; }
        }

        private void AnvikAnalysisForm_Load(object sender, EventArgs e)
        {
            serverGroupComboBox.Items.Clear();
            foreach (ServerGroup sg in problem.ServerGroups) {
                serverGroupComboBox.Items.Add(sg.Name);
            }
            systemStatesTextBox.Text = analysis.GetSystemStates().ToString();
            actionsTextBox.Text = analysis.GetActions().ToString();
            UpdateServerGroupGroupBox();
            optimizerComboBox.SelectedIndex = 0;
            ignoreRevenueCheckBox.Enabled = true;
            allowRejectCheckBox.Enabled = true;
            strictConvCheckBox.Enabled = true;
            optimizeButton.Enabled = true;
            cancelButton.Enabled = false;
            optimizeStatusTextBox.Text = "Ready";
            objectiveTextBox.Text = "";
            viewPolicyButton.Enabled = false;
            savePolicyButton.Enabled = false;
        }

        private void serverGroupComboBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            UpdateServerGroupGroupBox();
        }

        private void UpdateServerGroupGroupBox()
        {
            int idx = serverGroupComboBox.SelectedIndex;
            if (idx >= 0)
            {
                serverStatesTextBox.Text = analysis.GetServerStates((uint)idx).ToString();
                groupStatesTextBox.Text = analysis.GetGroupStates((uint)idx).ToString();
            }
            else
            {
                serverStatesTextBox.Text = "";
                groupStatesTextBox.Text = "";
            }
        }

        private void optimizeButton_Click(object sender, EventArgs e)
        {
            Thread waitCompletionThread = new Thread(
                new ThreadStart(() =>
                {
                    MethodInvoker onOptimizationComplete = new MethodInvoker(OnOptimizationComplete);
                    Analysis.JoinOptimize();
                    optimizationRunning = false;
                    Invoke(onOptimizationComplete);
                }
            ));
            Thread updateStatusThread = new Thread(
                new ThreadStart(() =>
                {
                    MethodInvoker onOptimizationStatusUpdate = new MethodInvoker(OnOptimizationStatusUpdate);
                    while (Analysis.GetOptimizerState() == OptimizerState.Started) {
                        Invoke(onOptimizationStatusUpdate);
                        Thread.Sleep(500);
                    }
                }
            ));
            optimizationRunning = true;
            optimizationCanceled = false;
            ignoreRevenueCheckBox.Enabled = false;
            allowRejectCheckBox.Enabled = false;
            strictConvCheckBox.Enabled = false;
            optimizeButton.Enabled = false;
            cancelButton.Enabled = true;
            optimizeStatusTextBox.Text = "Starting...";
            objectiveTextBox.Text = "";
            savePolicyButton.Enabled = false;
            // start computation thread
            OptimizerVersion optVer;
            if (optimizerComboBox.SelectedIndex == 0)
            {
                optVer = OptimizerVersion.CpuOptimizer;
            }
            else
            {
                optVer = OptimizerVersion.GpuOptimizer;
            }
            try
            {
                Analysis.StartOptimize(optVer, ignoreRevenueCheckBox.Checked, allowRejectCheckBox.Checked, strictConvCheckBox.Checked);
                // start thread that joins
                waitCompletionThread.Start();
                // start thread that updates UI
                updateStatusThread.Start();
            }
            catch (OptimizerVersionNotAvailable err) 
            {
                optimizationRunning = false;
                optimizationCanceled = true;
                OnOptimizationComplete();
                MessageBox.Show(this, err.Message, "Optimizer not available", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
        }

        private void OnOptimizationStatusUpdate()
        {
            optimizeStatusTextBox.Text = "Iteration " + Analysis.GetIteration().ToString();
        }

        private void OnOptimizationComplete()
        {
            ignoreRevenueCheckBox.Enabled = true;
            allowRejectCheckBox.Enabled = true;
            strictConvCheckBox.Enabled = true;
            optimizeButton.Enabled = true;
            cancelButton.Enabled = false;
            if (!optimizationCanceled)
            {
                optimizeStatusTextBox.Text = "Completed (" + Analysis.GetIteration().ToString() + " iter.)";
                objectiveTextBox.Text = Analysis.GetBestObjective().ToString();
                viewPolicyButton.Enabled = true;
                savePolicyButton.Enabled = true;
            }
            else
            {
                optimizeStatusTextBox.Text = "Ready";
                viewPolicyButton.Enabled = false;
                savePolicyButton.Enabled = false;
            }
        }

        private void AnvikAnalysisForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (optimizationRunning)
            {
                MessageBox.Show("Cannot close dialog while optimizing", "Information", MessageBoxButtons.OK, MessageBoxIcon.Information);
                e.Cancel = true;
            }
        }

        private void cancelButton_Click(object sender, EventArgs e)
        {
            if (optimizationRunning)
            {
                optimizeStatusTextBox.Text = "Canceling...";
                optimizationCanceled = true; 
                Analysis.CancelOptimize();
            }
        }

        private void viewPolicyButton_Click(object sender, EventArgs e)
        {
            policyViewForm.Problem = problem;
            policyViewForm.Analysis = analysis;
            policyViewForm.ShowDialog(this);
        }

        private void savePolicyButton_Click(object sender, EventArgs e)
        {
            saveFileDialog1.Filter = "Anvik Policy File|*.aks";
            DialogResult result = saveFileDialog1.ShowDialog(this);
            if (result == System.Windows.Forms.DialogResult.OK)
            {
                String fname = saveFileDialog1.FileName;
                Analysis.SaveBestPolicy(fname);
            }
        }
    }
}
