using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Media;

namespace anvik.gui
{
    public partial class AnvikPolicyViewForm : Form
    {
        private class MessageFilter : IMessageFilter
        {
            public AnvikPolicyViewForm Main { get; set; }

            public bool PreFilterMessage(ref Message msg)
            {
                const int WM_KEYDOWN = 0x100;
                const int WM_KEYUP = 0x101;
                if (msg.Msg == WM_KEYDOWN)
                {
                    var keyData = (Keys)msg.WParam;
                    if (keyData == Keys.PageDown || keyData == Keys.PageUp)
                    {
                        return true;
                    }
                }
                else if (msg.Msg == WM_KEYUP)
                {
                    var keyData = (Keys)msg.WParam;
                    if (keyData == Keys.PageUp)
                    {
                        Main.TryPreviousState();
                        return true;
                    }
                    else if (keyData == Keys.PageDown)
                    {
                        Main.TryNextState();
                        return true;
                    }
                }
                return false;
            }
        }

        public AnvikPolicyViewForm()
        {
            InitializeComponent();
            Application.AddMessageFilter(new MessageFilter { Main = this });
        }

        private Analysis analysis;
        private Problem problem;
        private UInt64 state;
        private UInt32 action;

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

        private void AnvikPolicyViewForm_Load(object sender, EventArgs e)
        {
            ChangeState(0);
            vmClassComboBox.Items.Clear();
            foreach (VmClass vmc in Problem.VmClasses)
            {
                vmClassComboBox.Items.Add(vmc.Name);
            }
            vmClassComboBox.SelectedIndex = 0;
        }

        private void TryChangeState()
        {
            UInt64 state;
            if (UInt64.TryParse(stateNumberTextBox.Text, out state))
            {
                if (state >= 0 && state < Analysis.GetSystemStates())
                {
                    ChangeState(state);
                }
                else
                {
                    ShowStateInputError();
                }
            }
            else
            {
                ShowStateInputError();
            }
        }

        public void TryPreviousState()
        {
            if (state > 0)
            {
                ChangeState(state - 1);
            }
            else
            {
                SystemSounds.Beep.Play();
            }
        }

        public void TryNextState()
        {
            if (state + 1 < Analysis.GetSystemStates())
            {
                ChangeState(state + 1);
            }
            else
            {
                SystemSounds.Beep.Play();
            }
        }

        public void ChangeState(UInt64 next_state)
        {
            state = next_state;
            action = Analysis.GetBestPolicy(state);
            stateNumberTextBox.Text = state.ToString();
            actionNumberTextBox.Text = action.ToString();
            stateDescTextBox.Text = Analysis.DescribeState(state);
            actionDescTextBox.Text = Analysis.DescribeAction(action);

        }

        private void ShowStateInputError()
        {
            String msg = "Not a valid state numer: states 0-" + (Analysis.GetSystemStates() - 1);
            MessageBox.Show(this, msg, "Invalid Input", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
            stateNumberTextBox.Focus();
        }

        private void stateNumberTextBox_Validating(object sender, CancelEventArgs e)
        {
            TryChangeState();
        }

        private void stateNumberTextBox_KeyPress(object sender, KeyPressEventArgs e)
        {
            if (e.KeyChar == '\r')
            {
                TryChangeState();
            }
        }

        private void onArrivalButton_Click(object sender, EventArgs e)
        {
            int idx = vmClassComboBox.SelectedIndex;
            if (idx >= 0)
            {
                UInt32 vmClass = (UInt32)idx;
                UInt64 next_state = Analysis.ComputeDestinationOnArrival(state, action, vmClass);
                ChangeState(next_state);
            }
        }

        private void AnvikPolicyViewForm_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Down)
            {
                if (state + 1 < Analysis.GetSystemStates())
                {
                    ChangeState(state + 1);
                }
            }
        }

    }
}
