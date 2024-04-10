from mutation_models import codegen_mutate
from utils import read_python_file, extract_code_section, get_class
from evaluations import is_trainable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np
import os
import random
import hydra
import ray

from conf.config import Config


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1, 1)
        self.fc1 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x


def fixer_test(cfg):
    ray.init()
    exp_name = "0003"
    path_nets = f"codegen_tests/{exp_name}"
    if not os.path.exists(path_nets):
        os.makedirs(os.path.normpath(path_nets))
    main_net_focus = read_python_file(os.path.normpath("/mnt/lustre/users/mnasir/NAS-LLM/best_net.py"))
    print(main_net_focus)
    #fixing_prompt = '"""Create another class that inherits from nn.Module to edit the above network such that it works when torch.zeros((1, 3, 32, 32)) is passed as an input"""'
    #fixing_prompt = '"""Fix the above neural network bu editing the layers such that it takes torch.zeros((1,3,32,32)) as inputs and output as tensor of size torch.size((1,10))"""'
    fixing_prompt = '"""The above neural network does not work in the current form. Add or delete layers to fix the above neural network such that it takes torch.zeros((1,3,32,32)) as inputs and output as tensor of size torch.size((1,10))"""'
    for i in range(10):
        (print(f"Test {i}"))
        
        fixed_net_results = codegen_mutate.remote(cfg=cfg, prompt=main_net_focus + "\n" + fixing_prompt, temperature = 0.1)
        extract_code_section(ray.get(fixed_net_results), fixing_prompt, file_path=os.path.normpath(f"{path_nets}/fixed_net_{i}.py"))
        print(read_python_file(os.path.normpath(f"{path_nets}/fixed_net_{i}.py")))
        try:
            Net = get_class(os.path.normpath(f"{path_nets}/fixed_net_{i}.py"))             
            net = Net()
            if is_trainable(net):
                print("TRUE")
            else:
                print("FALSE")
        except Exception as e:
            print(f"FALSE because of {e}")


def init_net_test(cfg):
    ray.init()
    exp_name = "init_net_test_0007"
    path_nets = f"codegen_tests/{exp_name}"
    if not os.path.exists(path_nets):
        os.makedirs(os.path.normpath(path_nets))
    init_net_prompt = '"""Create a simple neural network class that inherits from nn.Module pytorch class. It should accept input tensor of size 32 x 32 x 3 and output 10 neurons for classification task"""'
    for i in range(10):
        (print(f"Test {i}"))
        
        fixed_net_results = codegen_mutate.remote(cfg=cfg, prompt=init_net_prompt, temperature = 0.3)
        extract_code_section(ray.get(fixed_net_results), init_net_prompt, file_path=os.path.normpath(f"{path_nets}/init_net_{i}.py"))
        print(read_python_file(os.path.normpath(f"{path_nets}/init_net_{i}.py")))
        try:
            Net = get_class(os.path.normpath(f"{path_nets}/init_net_{i}.py"))             
            net = Net()
            if is_trainable(net):
                print("TRUE")
            else:
                print("FALSE")
        except Exception as e:
            print(f"FALSE because of {e}")

def mutation_test(cfg):
    ray.init()
    exp_name = "mutation_test_0005"

    prompts = ['"""Add a layer to improve the above network"""',
               '"""Delete a layer to improve the above network"""',
               '"""Increase the width of the above neural network"""',
               '"""Decrease the width of the above neural network"""',
               '"""Increase the depth of the above neural network"""',
               '"""Decrease the depth of the above neural network"""',
               '"""Add fully connected layer to improve the above network"""',
               '"""Add convolutional layer to improve the above network"""',
               '"""Add pooling layer to improve the above network"""',
               '"""Add residual connection to improve the above network"""',
               '"""Add multiple residual connections to improve the above network"""',
               '"""Add dropout layer to improve the above network"""',
               '"""Add normalization layer to improve the above network"""',
                ]
    path_nets = f"codegen_tests/{exp_name}"
    if not os.path.exists(path_nets):
        os.makedirs(os.path.normpath(path_nets))

    
    seed_value = 1
    for prompt in prompts:
        (print(f"Test on prompt: {prompt}"))

        seed_value = seed_value + 10
        torch.manual_seed(seed_value)

        mutation_prompt = \
        f'''class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1, 1)
        self.fc1 = nn.Linear(1024, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x
    
{prompt}'''
        
        fixed_net_results = codegen_mutate.remote(cfg=cfg, prompt=mutation_prompt, temperature = 0.6)
        extract_code_section(ray.get(fixed_net_results), mutation_prompt, file_path=os.path.normpath(f"{path_nets}/mu_net_{prompt}.py"))
        print(read_python_file(os.path.normpath(f"{path_nets}/mu_net_{prompt}.py")))
        try:
            Net = get_class(os.path.normpath(f"{path_nets}/mu_net_{prompt}.py"))             
            net = Net()
            if is_trainable(net):
                print("TRUE")
            else:
                print("FALSE")
        except Exception as e:
            print(f"FALSE because of {e}")


def crossover_test(cfg):
    ray.init()
    exp_name = "crossover_test_0001"

    prompt = '''

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    class Net2(nn.Module):
        def __init__(self):
            super(Net2, self).__init__()
            self.conv1 = nn.Conv2d(3, 1, 1)
            self.fc1 = nn.Linear(1024 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 10)
    
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x
    
    """Perform a crossover between above two neural networks and create a third neural network class that gets the best layers from above two networks"""'''
    path_nets = f"codegen_tests/{exp_name}"
    if not os.path.exists(path_nets):
        os.makedirs(os.path.normpath(path_nets))
    for i in range(10):
        (print(f"Test {i}"))
        
        fixed_net_results = codegen_mutate.remote(cfg=cfg, prompt=prompt, temperature = 0.6)
        extract_code_section(ray.get(fixed_net_results), prompt, file_path=os.path.normpath(f"{path_nets}/init_net_{i}.py"))
        print(read_python_file(os.path.normpath(f"{path_nets}/init_net_{i}.py")))
        try:
            Net = get_class(os.path.normpath(f"{path_nets}/init_net_{i}.py"))             
            net = Net()
            if is_trainable(net):
                print("TRUE")
            else:
                print("FALSE")
        except Exception as e:
            print(f"FALSE because of {e}")


def diff_mutate_test(cfg):
    ray.init()
    exp_name = "diff_mutate_test_0004"
    path_nets = f"codegen_tests/{exp_name}"
    if not os.path.exists(path_nets):
        os.makedirs(os.path.normpath(path_nets))
    diff_prompts = ['<NME> initial_net.py\n'
    '<BFE> import torch\n'
    'import torch.nn as nn\n'
    'import torch.nn.functional as F\n'
    '"""Returns a pytorch neural network class that takes an image of 3 x 32 x 32 as input and outputs 10 neurons."""\n'
    'class Net(nn.Module):\n'
    '   def __init__(self):\n'
    '       super().__init__()\n'
    '       self.conv1 = nn.Conv2d(3, 1, 1)\n'
    '       self.fc1 = nn.Linear(1024, 10)\n'
    '   def forward(self, x):\n'
    '       x = F.relu(self.conv1(x))\n'
    '       x = torch.flatten(x, 1)\n'
    '       x = F.relu(self.fc1(x))\n'
    '       return x\n'
    '<MSG> Added a nn.Conv2d layer to improve the neural network.\n',
    '<NME> initial_net.py\n'
    '<BFE> import torch\n'
    'import torch.nn as nn\n'
    'import torch.nn.functional as F\n'
    '"""Returns a pytorch neural network class that takes an image of 3 x 32 x 32 as input and outputs 10 neurons."""\n'
    'class Net(nn.Module):\n'
    '   def __init__(self):\n'
    '       super().__init__()\n'
    '       self.conv1 = nn.Conv2d(3, 1, 1)\n'
    '       self.fc1 = nn.Linear(1024, 10)\n'
    '   def forward(self, x):\n'
    '       x = F.relu(self.conv1(x))\n'
    '       x = torch.flatten(x, 1)\n'
    '       x = F.relu(self.fc1(x))\n'
    '       return x\n'
    '<MSG> Added a nn.Linear layer to improve the neural network.\n',
    '<NME> initial_net.py\n'
    '<BFE> import torch\n'
    'import torch.nn as nn\n'
    'import torch.nn.functional as F\n'
    '"""Returns a pytorch neural network class that takes an image of 3 x 32 x 32 as input and outputs 10 neurons."""\n'
    'class Net(nn.Module):\n'
    '   def __init__(self):\n'
    '       super().__init__()\n'
    '       self.conv1 = nn.Conv2d(3, 1, 1)\n'
    '       self.fc1 = nn.Linear(1024, 10)\n'
    '   def forward(self, x):\n'
    '       x = F.relu(self.conv1(x))\n'
    '       x = torch.flatten(x, 1)\n'
    '       x = F.relu(self.fc1(x))\n'
    '       return x\n'
    '<MSG> Added a nn.Conv2d and a nn.Linear layer to improve the neural network.\n',
    '<NME> initial_net.py\n'
    '<BFE> import torch\n'
    'import torch.nn as nn\n'
    'import torch.nn.functional as F\n'
    '"""Returns a pytorch neural network class that takes an image of 3 x 32 x 32 as input and outputs 10 neurons."""\n'
    'class Net(nn.Module):\n'
    '   def __init__(self):\n'
    '       super().__init__()\n'
    '       self.conv1 = nn.Conv2d(3, 1, 1)\n'
    '       self.fc1 = nn.Linear(1024, 10)\n'
    '   def forward(self, x):\n'
    '       x = F.relu(self.conv1(x))\n'
    '       x = torch.flatten(x, 1)\n'
    '       x = F.relu(self.fc1(x))\n'
    '       return x\n'
    '<MSG> Added layers to improve the neural network.\n']
    for i,diff_prompt in enumerate(diff_prompts):
        (print(f"Test {i}"))
        
        fixed_net_results = codegen_mutate.remote(cfg=cfg, prompt=diff_prompt, temperature = 0.8)
        res = ray.get(fixed_net_results)
        print(res)
        extract_code_section(res, diff_prompt, file_path=os.path.normpath(f"{path_nets}/diff_prompt_{i}.py"))
        print(read_python_file(os.path.normpath(f"{path_nets}/diff_prompt_{i}.py")))
        try:
            Net = get_class(os.path.normpath(f"{path_nets}/diff_prompt_{i}.py"))             
            net = Net()
            if is_trainable(net):
                print("TRUE")
            else:
                print("FALSE")
        except Exception as e:
            print(f"FALSE because of {e}")


@hydra.main(version_base="1.3.0", config_path="conf", config_name="config")    
def main(cfg: Config):
    #fixer_test(cfg)
    #init_net_test(cfg)
    #mutation_test(cfg)
    crossover_test(cfg)
    #diff_mutate_test(cfg)


if __name__ == "__main__":
    main()
