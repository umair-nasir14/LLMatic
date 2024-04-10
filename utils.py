import matplotlib.pyplot as plt
import ast
import importlib.util
import inspect
import os
import csv
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_network_width_depth_ratio(net):
    depth = 0
    width = []
    for name, module in net.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                               nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
                               nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                               nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                               nn.LocalResponseNorm,
                               nn.Linear, nn.Bilinear,
                               nn.Dropout, nn.Dropout2d, nn.Dropout3d,
                               nn.Embedding, nn.EmbeddingBag,
                               nn.LSTM, nn.GRU, nn.RNN,
                               nn.PReLU, nn.ReLU, nn.ReLU6, nn.RReLU,
                               nn.SELU, nn.CELU, nn.ELU, nn.GELU, nn.SiLU,
                               nn.Sigmoid, nn.Tanh, nn.LogSigmoid, nn.Softplus, nn.Softshrink,
                               nn.Softsign, nn.Tanhshrink, nn.Threshold,
                               nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
                               nn.AdaptiveLogSoftmaxWithLoss, nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d,
                               nn.AdaptiveMaxPool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
                               nn.FractionalMaxPool2d, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
                               nn.LPPool1d, nn.LPPool2d, nn.LocalResponseNorm, nn.Softmax, nn.Softmin,
                               nn.LogSoftmax, nn.Threshold)):
            depth += 1
            #if hasattr(module, 'out_features'):
            if isinstance(module,nn.Conv2d):
                width.append(module.out_channels)
            elif hasattr(module, 'out_features'):
                width.append(module.out_features)
    ratio = max(width)/depth
    print(width)
    print(depth)
    return ratio



def count_parameters(model):
    """
    Counts the number of parameters in a PyTorch model.
    
    Args:
        model: PyTorch model.
    
    Returns:
        None.
    """
    total_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
    return total_params

class Net():
    """Dummy class. Will be replaced by evolved networks."""
    pass

def get_max_curiosity(species_list, n):
    species_list_sorted = sorted(species_list, key=lambda x: x.curiosity, reverse=True)
    return species_list_sorted[:n]

def get_max_fitness(species_list, n):
    species_list_sorted = sorted(species_list, key=lambda x: x.fitness, reverse=False)
    return species_list_sorted[:n]


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def read_python_file(filepath):
    remove_after_return(filepath)
    with open(filepath, "r") as f:
        code = f.read()
    return code

def extract_code(generation, prompt, file_path):
    code_parts = generation.split(prompt)

    with open(file_path, "w") as f:
        f.write(code_parts[1])


def get_net_name(filename):
    with open(filename, 'r') as file:
        source = file.read()

    class_nodes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)]
    if len(class_nodes) > 0:
        return class_nodes[0].name
    else:
        return None

def extract_code_section(code_string: str, prompt, file_path):
    # Split the input string into code sections based on commented lines
    current_section = []
    code_section = " "
    #print(f"code after mutation:\n {code_string}")
    for line in code_string.splitlines():
        if line.strip().startswith('"""'):
            # Start a new code section
            if current_section:
                code_section = "\n".join(current_section)
                current_section = []
        else:
            # Add the line to the current code section
            current_section.append(line)

    with open(file_path, "w") as f:
        f.write(f"import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n{code_string}")



def get_class(filename):
    module = os.path.basename(filename)
    spec = importlib.util.spec_from_file_location(module[:-3], filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            return obj
        
    return None


def csv_writer(results, output_file):

    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(results)

def count_unique_components(net):
    # Initialize a set to store the types of all components
    component_types = set()
    
    # Traverse the network to extract the type of each component
    for module in net.modules():
        module_type = type(module).__name__
        if module_type not in ["Sequential", "ModuleList", "ModuleDict", "Tensor"]:
            component_types.add(module_type)
    
    # Count the number of unique component types
    num_unique_components = len(component_types)
    
    return num_unique_components

def remove_after_return(file_path):
    # Read in the file
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Find the index of the first "return" statement
    for i, line in enumerate(lines):
        if "return" in line:
            break

    # Remove everything after the "return" statement
    new_lines = lines[:i+1]

    # Write the updated code to the file
    with open(file_path, "w") as f:
        f.writelines(new_lines)


