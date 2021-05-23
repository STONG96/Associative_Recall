# =====================
# Original Source: https://github.com/vlgiitr/ntm-pytorch/tree/master/ntm/datasets
# =====================
import torch
from torch import nn
import torch.nn.functional as F


class Controller(nn.Module):

    def __init__(self, input_size, controller_size, output_size, read_data_size):
        super(Controller, self).__init__()

        self.input_size = input_size
        self.ctrl_dim = controller_size
        self.output_size = output_size
        self.read_data_size = read_data_size

        # Controller neural network
        self.controller_net = nn.LSTMCell(input_size, controller_size)
        # Output neural network
        self.out_net = nn.Linear(read_data_size, output_size)
        # Initialize the weights of output net
        nn.init.kaiming_uniform_(self.out_net.weight)

        # Learnable initial hidden and cell states
        self.h_state = torch.zeros([1, controller_size])
        self.c_state = torch.zeros([1, controller_size])
        # Layers to learn init values for controller hidden and cell states
        self.h_bias_fc = nn.Linear(1, controller_size)
        self.c_bias_fc = nn.Linear(1, controller_size)
        # Reset
        self.reset()

    def forward(self, in_data, prev_reads):
        '''Returns the hidden and cell states'''
        x = torch.cat([in_data] + prev_reads, dim=-1)
        # Get hidden and cell states
        self.h_state, self.c_state = self.controller_net(x, (self.h_state, self.c_state))

        return self.h_state, self.c_state

    def output(self, read_data):
        '''Returns the external output from the read vectors'''
        complete_state = torch.cat([self.h_state] + read_data, dim=-1)
        # Compute output
        output = F.sigmoid(self.out_net(complete_state))

        return output

    def reset(self, batch_size=1):
        '''Reset/initialize the controller states'''
        # Dummy input
        in_data = torch.tensor([[0.]])
        # Hidden state
        h_bias = self.h_bias_fc(in_data)
        self.h_state = h_bias.repeat(batch_size, 1)
        # Cell state
        c_bias = self.c_bias_fc(in_data)
        self.c_state = c_bias.repeat(batch_size, 1)