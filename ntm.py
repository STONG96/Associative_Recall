# =====================
# Original Source: https://github.com/vlgiitr/ntm-pytorch/tree/master/ntm/datasets
# =====================
import torch
from torch import nn
from modules import Memory, Head, Controller


class NTM(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 controller_size,
                 memory_units,
                 memory_unit_size,
                 num_heads):
        super(NTM, self).__init__()

        # Create controller
        self.controller_size = controller_size
        self.controller = Controller(input_size + num_heads * memory_unit_size, controller_size, output_size,
            read_data_size=controller_size + num_heads * memory_unit_size)

        # Create memory
        self.memory = Memory(memory_units, memory_unit_size)
        self.memory_unit_size = memory_unit_size  # M
        self.memory_units = memory_units  # N

        # Create Heads
        self.num_heads = num_heads
        self.heads = nn.ModuleList([])
        for head in range(num_heads):
            self.heads += [
                Head('r', controller_size, memory_unit_size),
                Head('w', controller_size, memory_unit_size)
            ]

        # Init previous head weights and read vectors
        self.prev_head_weights = []
        self.prev_reads = []
        # Layers to initialize the weights and read vectors
        #self.head_weights_fc = nn.Linear(1, self.memory_units)
        #self.reads_fc = nn.Linear(1, self.memory_unit_size)

        self.reset()

    def reset(self, batch_size=1):
        self.memory.reset(batch_size)
        self.controller.reset(batch_size)
        self.prev_head_weights = []
        for i in range(len(self.heads)):
            prev_weight = torch.zeros([batch_size, self.memory.n])
            self.prev_head_weights.append(prev_weight)
        self.prev_reads = []
        for i in range(self.num_heads):
            prev_read = torch.zeros([batch_size, self.memory.m])
            # using random initialization for previous reads
            nn.init.kaiming_uniform_(prev_read)
            self.prev_reads.append(prev_read)

    def forward(self, in_data):
        controller_h_state, controller_c_state = self.controller(
            in_data, self.prev_reads)
        read_data = []
        head_weights = []
        for head, prev_head_weight in zip(self.heads, self.prev_head_weights):
            if head.mode == 'r':
                head_weight, r = head(
                    controller_c_state, prev_head_weight, self.memory)
                read_data.append(r)
            else:
                head_weight, _ = head(
                    controller_c_state, prev_head_weight, self.memory)
            head_weights.append(head_weight)

        output = self.controller.output(read_data)

        self.prev_head_weights = head_weights
        self.prev_reads = read_data

        return output

