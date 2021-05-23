# Neural Turing Machines (Pytorch) #

Code for the task Associative Recall in paper

[Neural Turing Machines](https://arxiv.org/pdf/1410.5401)

Alex Graves, Greg Wayne, Ivo Danihelka


## Requirements

- PyTorch
- numpy
- tqdm
- os
- tensorboard_logger
- argparse
- matplotlib


## Usage

1. Run the train.py script for training the model with default settings.
2. From a terminal, navigate to the repo. 
Run the command below to get Loss and Errors.
    ```
    tensorboard --logdir=ntmrun
    ```
1.  Run the visualization.py for test, evaluate and visualization.


## Credits
- Neural Turing Machines https://arxiv.org/abs/1410.5401
- Code is based on https://github.com/nerdimite/ntm
- Implementing Neural Turing Machines https://arxiv.org/abs/1807.08518