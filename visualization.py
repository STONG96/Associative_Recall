# =====================
# Original Source: https://github.com/vlgiitr/ntm-pytorch/tree/master/ntm/datasets
# =====================
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import torch
from torch import nn, optim
from args import get_parser
from ntm import NTM
from ARtask import AssociativeDataset


# Load the Task Configuration files
args = get_parser().parse_args()
args.task_json = 'configs/associative.json'
task_params = json.load(open(args.task_json))
criterion = nn.BCELoss()

# Evaluation parameters for AssociativeRecall task
task_params['min_item'] = 6
task_params['max_item'] = 20

# Dataset
dataset = AssociativeDataset(task_params)
args.saved_model = 'saved_model_associative.pt'
cur_dir = os.getcwd()
PATH = os.path.join(cur_dir, args.saved_model)


# Create NTM
ntm = NTM(input_size=task_params['seq_width'] + 2,
          output_size=task_params['seq_width'],
          controller_size=task_params['controller_size'],
          memory_units=task_params['memory_units'],
          memory_unit_size=task_params['memory_unit_size'],
          num_heads=task_params['num_heads'])

# Load trained model weights
ntm.load_state_dict(torch.load(PATH))

# Reset
ntm.reset()

# Sample data
data = dataset[0]  # 0 is a dummy index
input, target = data['input'], data['target']

# Tensor to store outputs
out = torch.zeros(target.size())

# Process the inputs through NTM for memorization
for i in range(input.size()[0]):
    # to maintain consistency in dimensions as torch.cat was throwing error
    in_data = torch.unsqueeze(input[i], 0)
    ntm(in_data)

# passing zero vector as the input while generating target sequence
in_data = torch.unsqueeze(torch.zeros(input.size()[1]), 0)
for i in range(target.size()[0]):
    out[i] = ntm(in_data)


# Loss and error
loss = criterion(out, target)

# Calculated binary outputs
binary_output = out.clone()
binary_output = binary_output.detach().apply_(lambda x: 0 if x < 0.5 else 1)

# Sequence prediction error is calculated in bits per sequence
error = torch.sum(torch.abs(binary_output - target))

# Logging
print('Loss: %.2f\tError in bits per sequence: %.2f' % (loss, error))

# Saving results
result = {'output': binary_output, 'target': target}


# Visualize input target and output
target_pic = target.numpy()
target_pic.shape
output_pic = binary_output.numpy()
output_pic.shape
input_pic = input.numpy()
input_pic.shape

plt.subplot(131)
plt.title('input')
plt.imshow(input_pic,cmap='gray')
plt.subplot(132)
plt.title('target')
plt.imshow(target_pic,cmap='gray')
plt.subplot(133)
plt.title('output')
plt.imshow(output_pic,cmap='gray')
plt.legend(loc='best')
plt.show()


