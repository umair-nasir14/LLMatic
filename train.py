import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp

from conf.config import Config
from datasets import get_datasets
import ray
import time

cfg = Config()

@ray.remote(num_gpus=cfg.NUM_GPUS)
def train_net_on_gpu(net, epochs=1):
    

    trainset, trainloader, validset, validloader = get_datasets()

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"The network will train on {dev} device")
    criterion = nn.CrossEntropyLoss()
    #breakpoint()
    optimizer = optim.SGD(net.to(dev).parameters(), lr=0.001, momentum=0.9)
    print("------TRAINING------")
    #net.train()
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        correct = 0.0
        
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(dev), labels.to(dev)
      
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = net(inputs).to(dev)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            correct += (outputs == labels).float().sum()
        training_accuracy = 100 * correct / len(trainset)
        
        valid_loss = 0.0
        valid_correct = 0.0
        net.eval()     # Optional when not using Model Specific layer
        for i, d in enumerate(validloader):
            # Transfer Data to GPU if available
            data, labels = d
            data, labels = data.to(dev), labels.to(dev)
            target = net(data).to(dev)
            # Find the Loss
            loss = criterion(target,labels)
            # Calculate Loss
            valid_loss += loss.item()
            valid_correct += (target == labels).float().sum()
        valid_accuracy = 100 * valid_correct / len(validset)
        print(f'Training Epochs: {epoch}\t\t Loss: {running_loss}\t\t Train Accuracy: {training_accuracy}')
        print(f'\t\t Valid Loss: {valid_loss}\t\t Valid Accuracy: {valid_accuracy}')
            
    return valid_loss

@ray.remote(num_cpus=cfg.NUM_CPUS)
def train_net_on_cpu(net, epochs=1):
    

    trainset, trainloader, validset, validloader = get_datasets()

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"The network will train on {dev} device")
    criterion = nn.CrossEntropyLoss()
    #breakpoint()
    optimizer = optim.SGD(net.to(dev).parameters(), lr=0.001, momentum=0.9)
    print("------TRAINING------")
    #net.train()
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        correct = 0.0
        training_loss = 0.0
        valid_loss = 0.0
        valid_correct = 0.0
        valid_running_loss = 0.0
        total = 0.0
        val_total = 0.0
        
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(dev), labels.to(dev)
      
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = net(inputs).to(dev)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            training_loss += loss.item()
            total += labels.size(0)
            correct += (outputs == labels).float().sum()
            if i % 2000 == 1999:
                #print('[%d, %5d] loss: %.3f, accuracy: %.3f' %
                #      (epoch + 1, i + 1, running_loss / 2000, correct / total))
                running_loss = 0.0
        training_accuracy = 100 * correct / total
        
        
        net.eval()     # Optional when not using Model Specific layer
        for i, d in enumerate(validloader):
            # Transfer Data to GPU if available
            data, labels = d
            data, labels = data.to(dev), labels.to(dev)
            target = net(data).to(dev)
            # Find the Loss
            loss = criterion(target,labels)
            # Calculate Loss
            valid_running_loss += loss.item()
            valid_loss += loss.item()
            val_total += labels.size(0)
            valid_correct += (target == labels).float().sum()

            if i % 2000 == 1999:
                #print('[%d, %5d] loss: %.3f, accuracy: %.3f' %
                #      (epoch + 1, i + 1, running_loss / 2000, valid_correct / val_total))
                valid_running_loss = 0.0
        valid_accuracy = 100 * valid_correct / val_total
        print(f'Training Epochs: {epoch}\t\t Training Loss: {training_loss / len(trainloader)}\t\t Train Accuracy: {training_accuracy}')
        print(f'\t\t Valid Loss: {valid_loss / len(validloader)}\t\t Valid Accuracy: {valid_accuracy}')
            
    return valid_loss / len(validloader)



@ray.remote(num_cpus=cfg.NUM_CPUS)
def forward_pass_on_cpu(net):
    trainset, trainloader, validset, validloader = get_datasets()

    # Define model architecture

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"The network will train on {dev} device")
    print("------Forward passing------")
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    #breakpoint()
    optimizer = optim.SGD(net.to(dev).parameters(), lr=0.001, momentum=0.9)
    batches = 0
    running_loss = 0
    correct = 0.0
    
    start = time.time()
    # Perform forward pass
    for batch_idx, (data, target) in enumerate(trainloader):
        # Set the gradients to zero
        optimizer.zero_grad()

        # Forward pass
        output = net(data).to(dev)

        # Compute loss
        loss = criterion(output, target)
        running_loss += loss.item()
        correct += (target == output).float().sum()
        
        #if batches == 2:
        #    break
        #batches +=1
        # Print loss every 1000 batches
        #if batch_idx % 1000 == 0:
        #    print('Batch Index : {} Loss : {}'.format(batch_idx, running_loss))
    inference_time = time.time() - start
    accuracy = 100 * correct / len(trainset)
    print(f'\t\t Loss: {running_loss/len(validloader)}\t\t Accuracy: {accuracy}\t\t Inference Time: {inference_time}')
    
    return running_loss / len(validloader), inference_time


@ray.remote(num_gpus=cfg.NUM_GPUS)
def forward_pass_on_gpu(net):
    trainset, trainloader, validset, validloader = get_datasets()

    # Define model architecture

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"The network will train on {dev} device")
    print("------Forward passing------")
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    #breakpoint()
    optimizer = optim.SGD(net.to(dev).parameters(), lr=0.001, momentum=0.9)
    batches = 0
    running_loss = 0
    correct = 0.0
    
    start = time.time()
    # Perform forward pass
    for batch_idx, (data, target) in enumerate(trainloader):
        # Set the gradients to zero
        optimizer.zero_grad()

        # Forward pass
        output = net(data).to(dev)

        # Compute loss
        loss = criterion(output, target)
        running_loss += loss.item()
        correct += (target == output).float().sum()
        
        #if batches == 2:
        #    break
        #batches +=1
        # Print loss every 1000 batches
        #if batch_idx % 1000 == 0:
        #    print('Batch Index : {} Loss : {}'.format(batch_idx, running_loss))
    inference_time = time.time() - start
    accuracy = 100 * correct / len(trainset)
    print(f'\t\t Loss: {running_loss/len(validloader)}\t\t Accuracy: {accuracy}\t\t Inference Time: {inference_time}')
    
    return running_loss / len(validloader), inference_time


def multiprocess_training(training_func, num_processes, *args, **kwargs):
    
    
    # Define a function to apply the training function to a set of arguments.
    #def process_training(args):
    #    return training_func(*args)

    # Define a multiprocessing pool with the desired number of processes.
    pool = mp.Pool(num_processes)

    # Apply the PyTorch training function to each set of arguments using the multiprocessing pool.
    results = pool.map(training_func, zip(*args))

    # Close the multiprocessing pool and join the processes.
    pool.close()
    pool.join()

    return results



def transfer_weights(netA, netB, layer_type):
    
    # Get the parameters of the two networks
    paramsA = netA.state_dict()
    paramsB = netB.state_dict()
    
    # Transfer the weights from the layers of the given type in netA to the corresponding layers in netB
    for name, module in netA.named_modules():
        if type(module).__name__ == layer_type:
            #print(f"paramsB[name + '.weight'].shape: {len(paramsB[name + '.bias'].shape)}")
            if len(paramsB[name + '.bias'].shape) == 5:
                paramsB[name + '.weight'][:, :, :, :, :] = paramsA[name + '.weight'][:, :, :, :, :]
            elif len(paramsB[name + '.bias'].shape) == 4:
                paramsB[name + '.weight'][:, :, :, :] = paramsA[name + '.weight'][:, :, :, :]
            elif len(paramsB[name + '.bias'].shape) == 3:
                paramsB[name + '.weight'][:, :, :] = paramsA[name + '.weight'][:, :, :]
            elif len(paramsB[name + '.bias'].shape) == 2:
                paramsB[name + '.weight'][:, :] = paramsA[name + '.weight'][:, :]
            elif len(paramsB[name + '.bias'].shape) == 1:
                paramsB[name + '.weight'][:] = paramsA[name + '.weight'][:]
            paramsB[name + '.bias'][:] = paramsA[name + '.bias'][:]
    
    # Set the state dict of netB to the updated parameters
    netB.load_state_dict(paramsB)

    return netB


def detect_layers(model):
    layers = []

    def detect_layers_recursively(module):
        for child in module.children():
            if isinstance(child, nn.Sequential):
                detect_layers_recursively(child)
            elif isinstance(child, nn.Module):
                layers.append(child.__class__.__name__)
                detect_layers_recursively(child)

    detect_layers_recursively(model)

    return list(set(layers))