# =====================
# Original Source: https://github.com/vlgiitr/ntm-pytorch/tree/master/ntm/datasets
# =====================
import json
from tqdm import tqdm
import numpy as np
import os

import torch
from torch import nn, optim
from tensorboard_logger import configure, log_value

from ntm import NTM
from ARtask import AssociativeDataset
from args import get_parser


args = get_parser().parse_args()
configure("ntmrun/")
# ----------------------------------------------------------------------------
# -- initialize datasets, model, criterion and optimizer
# ----------------------------------------------------------------------------
args.task_json = 'configs/associative.json'

# ==== Create Dataset ====
task_params = json.load(open(args.task_json))
dataset = AssociativeDataset(task_params)

# ==== Create NTM ====
ntm = NTM(input_size=task_params['seq_width'] + 2,
          output_size=task_params['seq_width'],
          controller_size=task_params['controller_size'],
          memory_units=task_params['memory_units'],
          memory_unit_size=task_params['memory_unit_size'],
          num_heads=task_params['num_heads'])

# ==== Training Settings ====
# Loss Function
criterion = nn.BCELoss()
# As the learning rate is task specific, the argument can be moved to json file
optimizer = optim.RMSprop(ntm.parameters(),
                          lr=args.lr,
                          alpha=args.alpha,
                          momentum=args.momentum)
'''
optimizer = optim.Adam(ntm.parameters(), lr=args.lr,
                       betas=(args.beta1, args.beta2))
'''

args.saved_model = 'saved_model_associative.pt'


cur_dir = os.getcwd()
PATH = os.path.join(cur_dir, args.saved_model)

# Training
losses = []
errors = []

for iter in tqdm(range(args.num_iters)):
    optimizer.zero_grad()
    ntm.reset()

    # Sample data
    data = dataset[iter]
    input, target = data['input'], data['target']

    # Tensor to store outputs
    out = torch.zeros(target.size())

    # Process the inputs through NTM for memorization
    for i in range(input.size()[0]):
        # to maintain consistency in dimensions as torch.cat was throwing error
        in_data = torch.unsqueeze(input[i], 0)
        ntm(in_data)

    # passing zero vector as input while generating target sequence
    in_data = torch.unsqueeze(torch.zeros(input.size()[1]), 0)
    for i in range(target.size()[0]):
        out[i] = ntm(in_data)

    # Compute loss, backprop and optimize
    loss = criterion(out, target)
    losses.append(loss.item())
    loss.backward()
    # clips gradient in the range [-10,10]. Again there is a slight but
    # insignificant deviation from the paper where they are clipped to (-10,10)
    nn.utils.clip_grad_value_(ntm.parameters(), 10)
    optimizer.step()

    # Calculate binary outputs
    binary_output = out.clone()
    binary_output = binary_output.detach().apply_(lambda x: 0 if x < 0.5 else 1)

    # Sequence prediction error is calculated in bits per sequence
    error = torch.sum(torch.abs(binary_output - target))
    errors.append(error.item())

    # Print Stats and tensorboard_logger
    if iter % 200 == 0:
        print('Iteration: %d\tLoss: %.2f\tError in bits per sequence: %.2f' %
              (iter, np.mean(losses), np.mean(errors)))
        log_value('train_loss', np.mean(losses), iter)
        log_value('bit_error_per_sequence', np.mean(errors), iter)
        losses = []
        errors = []

# Save model
torch.save(ntm.state_dict(), PATH)

